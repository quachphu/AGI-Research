            return Path(tempfile.gettempdir()) / f"triton-gsan-{token[:16]}-{rank}.sock"

        socket_path = make_socket_path(rank)
        socket_path.unlink(missing_ok=True)
        stack.callback(lambda: socket_path.unlink(missing_ok=True))

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stack.push(listener)
        listener.bind(str(socket_path))
        listener.listen(world_size)

        received_fds: dict[int, tuple[int, int, int | None]] = {}
        pending: dict[int, tuple[socket.socket, int, int, int, int | None]] = {}

        try:
            for peer in range(rank):
                while peer not in pending:
                    conn, _ = listener.accept()
                    src_rank, src_device_index, peer_real_fd, peer_shadow_fd, peer_runtime_state_fd = _recv_fds(conn)
                    if src_rank < 0 or src_rank >= world_size:
                        raise RuntimeError(f"Unexpected peer rank {src_rank}")
                    if int(metas[src_rank]["device_index"]) != src_device_index:
                        raise RuntimeError(f"Unexpected device index {src_device_index} from rank {src_rank}, "
                                           f"expected {metas[src_rank]['device_index']}")
                    pending[src_rank] = (conn, src_device_index, peer_real_fd, peer_shadow_fd, peer_runtime_state_fd)
                conn, _, peer_real_fd, peer_shadow_fd, peer_runtime_state_fd = pending.pop(peer)
                received_fds[peer] = (peer_real_fd, peer_shadow_fd, peer_runtime_state_fd)
                _send_fds(
                    conn,
                    src_rank=rank,
                    src_device_index=int(device_index),
                    real_fd=real_fd,
                    shadow_fd=shadow_fd,
                    runtime_state_fd=(runtime_state_fd if peer in peers_needing_runtime_bootstrap else None),
                )
                conn.close()
            for peer in range(rank + 1, world_size):
                peer_socket_path = make_socket_path(peer)
                # Connection is initiated by lower ranked peer
                with _connect(peer_socket_path) as sock:
                    _send_fds(
                        sock,
                        src_rank=rank,
                        src_device_index=int(device_index),
                        real_fd=real_fd,
                        shadow_fd=shadow_fd,
                        runtime_state_fd=(runtime_state_fd if peer in peers_needing_runtime_bootstrap else None),
                    )
                    src_rank, src_device_index, peer_real_fd, peer_shadow_fd, peer_runtime_state_fd = _recv_fds(sock)
                    if src_rank != peer:
                        raise RuntimeError(f"Unexpected peer rank {src_rank}, expected {peer}")
                    if int(metas[peer]["device_index"]) != src_device_index:
                        raise RuntimeError(f"Unexpected device index {src_device_index} from rank {src_rank}, "
                                           f"expected {metas[peer]['device_index']}")
                    received_fds[peer] = (peer_real_fd, peer_shadow_fd, peer_runtime_state_fd)
        except Exception:
            for peer_real_fd, peer_shadow_fd, peer_runtime_state_fd in received_fds.values():
                os.close(peer_real_fd)
                os.close(peer_shadow_fd)
                if peer_runtime_state_fd is not None:
                    os.close(peer_runtime_state_fd)
            received_fds.clear()
            raise
        finally:
            for conn, _, _, _, _ in pending.values():
                conn.close()

        peer_ptrs = _import_peer_ptrs(
            received_fds=received_fds,
            metas=metas,
            rank=rank,
            world_size=world_size,
            device_index=int(device_index),
            base_ptr=base_ptr,
        )
        dist.barrier(group=process_group)
        if peers_needing_runtime_bootstrap:
            runtime_bootstrapped_peers.update(peers_needing_runtime_bootstrap)

    handle = GSanSymmetricMemoryHandle(
        group=process_group,
        rank=rank,
        world_size=world_size,
        device_index=device_index,
        buffer_size=buffer_size,
        peer_ptrs=peer_ptrs,
        peer_device_indices=peer_device_indices,
        cache_key=cache_key,
    )
    _RENDEZVOUS_CACHE[cache_key] = handle
    return handle
<|endoftext|>from __future__ import annotations

from typing import Any

from triton._C.libtriton import gsan_testing as _gsan_testing

from triton._C.libtriton.gsan_testing import (
    ShadowCell,
    GlobalState,
    ThreadState,
    shadow_cell_address,
    thread_state_stride_bytes,
    decode_shadow_cell,
    SHADOW_CELL_SIZE_BYTES,
    GLOBAL_STATE_SIZE_BYTES,
)


def shadow_cell(real_address: int) -> tuple[int, int]:
    return shadow_cell_address(real_address), SHADOW_CELL_SIZE_BYTES


def decode_global_state(data: bytes | bytearray | memoryview) -> GlobalState:
    return _gsan_testing.decode_global_state(bytes(data))


def decode_global_state_tensor(state_bytes: Any) -> GlobalState:
    if hasattr(state_bytes, "detach"):
        tensor = state_bytes.detach().cpu().contiguous().view(-1)
        return decode_global_state(bytes(tensor[:GLOBAL_STATE_SIZE_BYTES].tolist()))
    return decode_global_state(bytes(state_bytes))


def decode_shadow_cell_tensor(cell_bytes: Any) -> ShadowCell:
    # Accept a torch.Tensor without importing torch at module import time.
    if hasattr(cell_bytes, "detach"):
        tensor = cell_bytes.detach().cpu().contiguous().view(-1)
        return decode_shadow_cell(bytes(tensor[:SHADOW_CELL_SIZE_BYTES].tolist()))
    return decode_shadow_cell(bytes(cell_bytes))


def decode_thread_state(data: bytes | bytearray | memoryview, num_threads: int, clock_buffer_size: int) -> ThreadState:
    return _gsan_testing.decode_thread_state(bytes(data), int(num_threads), int(clock_buffer_size))


def decode_thread_state_tensor(state_bytes: Any, num_threads: int, clock_buffer_size: int) -> ThreadState:
    if hasattr(state_bytes, "detach"):
        tensor = state_bytes.detach().cpu().contiguous().view(-1)
        size = thread_state_stride_bytes(num_threads, clock_buffer_size)
        return decode_thread_state(bytes(tensor[:size].tolist()), num_threads, clock_buffer_size)
    return decode_thread_state(bytes(state_bytes), num_threads, clock_buffer_size)
<|endoftext|>from __future__ import annotations
from triton.compiler.compiler import ASTSource
from triton.backends.compiler import Language
from triton.runtime.jit import JITFunction, constexpr_function
from typing import TypeVar, Optional, Callable, Iterable, Union, overload
from triton._C.libtriton import ir

T = TypeVar("T")

__all__ = ["GluonJITFunction", "constexpr_function", "jit"]


class GluonASTSource(ASTSource):

    def __init__(self, fn, signature, constexprs=None, attrs=None) -> None:
        super().__init__(fn, signature, constexprs, attrs)
        self.language = Language.GLUON
        self.ext = "ttgir"

    def make_ir(self, target, options, codegen_fns, module_map, context):
        from triton.compiler.compiler import make_backend
        from triton.compiler.code_generator import ast_to_ttir

        builder = ir.builder(context)
        module = builder.create_module()

        # Assign module attributes eagerly, as they are needed to verify layouts
        backend = make_backend(target)
        target = backend.get_target_name(options)

        module.set_attr("ttg.target", builder.get_string_attr(target))
        module.set_attr("ttg.num-warps", builder.get_int32_attr(options.num_warps))
        module.set_attr("ttg.num-ctas", builder.get_int32_attr(options.num_ctas))
        module.set_attr("ttg.threads-per-warp", builder.get_int32_attr(options.warp_size))

        is_cuda = options.backend_name == "cuda"
        if is_cuda and options.maxnreg is not None:
            module.set_attr("ttg.maxnreg", builder.get_int32_attr(options.maxnreg))

        module = ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                             module_map=module_map, module=module)
        return module


class GluonJITFunction(JITFunction[T]):

    def create_binder(self):
        result = super().create_binder()
        self.ASTSource = GluonASTSource
        return result

    def is_gluon(self):
        return True


@overload
def jit(fn: T) -> GluonJITFunction[T]:
    ...


@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], GluonJITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[GluonJITFunction[T], Callable[[T], JITFunction[T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        return GluonJITFunction(
            fn,
            version=version,
            do_not_specialize=do_not_specialize,
            do_not_specialize_on_alignment=do_not_specialize_on_alignment,
            debug=debug,
            noinline=noinline,
            repr=repr,
            launch_metadata=launch_metadata,
        )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
<|endoftext|>from ._runtime import GluonJITFunction, constexpr_function, jit
from triton import must_use_result, aggregate
from . import nvidia
from . import amd

__all__ = ["aggregate", "amd", "constexpr_function", "GluonJITFunction", "jit", "must_use_result", "nvidia"]
<|endoftext|>from typing import Sequence, List, TypeVar, Tuple, Callable
import math
from triton.language.semantic import TritonSemantic
from . import _core as ttgl
from ._layouts import AutoLayout, DistributedLayout, DistributedLinearLayout, SliceLayout, SharedLayout, CoalescedLayout, SharedLinearLayout
from triton._C.libtriton.gluon_ir import GluonOpBuilder, compute_tmem_reg_layout
from triton._C.libtriton import ir
from triton.compiler.code_generator import flatten_values_to_ir, unflatten_ir_values

TensorTy = TypeVar("TensorTy")


def _check(cond: bool, msg_fn: Callable[[], str], category=ValueError):
    if not cond:
        raise category(msg_fn())


def _is_int_list(value):
    return isinstance(value, Sequence) and all(isinstance(i, int) for i in value)


def _compute_tmem_reg_layout(element_ty, shape, alloc_shape, layout, num_warps, instr_variant):
    _check(isinstance(instr_variant, str), lambda: "instr_variant must be a string")
    _check(instr_variant in ("32x32b", "16x64b", "16x128b", "16x256b", "16x32bx2", "32x32b_splitn"),
           lambda: f"unknown instr_variant: {instr_variant}")
    _check(isinstance(num_warps, int), lambda: f"num_warps must be an int but got {type(num_warps)!r}")
    _check(num_warps >= 4 and (num_warps & (num_warps - 1)) == 0, lambda: "num_warps must be a power of two and >= 4")

    shape = list(shape)
    _check(all(isinstance(dim, int) for dim in shape), lambda: f"shape entries must be ints but got {shape}")
    rank = len(shape)
    _check(rank == 2, lambda: "expected a 2D tensor")
    alloc_shape = list(alloc_shape)
    _check(all(isinstance(dim, int) for dim in alloc_shape),
           lambda: f"alloc_shape entries must be ints but got {alloc_shape}")
    _check(len(alloc_shape) >= rank, lambda: f"alloc_shape must have rank >= shape rank, got {alloc_shape} and {shape}")

    splitn = instr_variant == "32x32b_splitn"
    atom_variant = "32x32b" if splitn else instr_variant

    layout_obj = compute_tmem_reg_layout(
        element_ty,
        shape,
        alloc_shape,
        layout,
        num_warps,
        atom_variant,
    )
    _check(layout_obj is not None,
           lambda: f"TMEM layout '{atom_variant}' unsupported for shape {shape} and num_warps {num_warps}")

    if splitn:
        N = shape[1]
        if not layout_obj.reg_bases:
            # We cannot use this layout in a load or a store ATM due to a PTX bug!
            # You can work around this by loading to 32x32b and follow by a convert_layout to this layout.
            _check(layout_obj.lane_bases[-1] == [0, N // 2],
                   lambda: f"splitn with 1 register requires the last lane basis to be [0, N / 2]. Got {layout_obj}")
            layout_obj.reg_bases.append([0, N // 2])
            layout_obj.lane_bases[-1] = [0, 0]
        elif layout_obj.reg_bases[-1] != [0, N // 2]:
            bitwidth = element_ty.primitive_bitwidth
            num_reg = 2**len(layout_obj.reg_bases)
            _check(
                num_reg > 32 // bitwidth, lambda: "To be able to `tmem.load` into `tl.split` you need to have more "
                f"than {32 // bitwidth} {bitwidth}-bit registers, as you need to use "
                "the instruction 32x32b.x1 twice. You can always load into "
                "instr_variant=\"32x32b\" and then convert_layout to this layout otherwise.")

            reg_bases = layout_obj.reg_bases
            for bases_str in ("lane_bases", "warp_bases"):
                bases = getattr(layout_obj, bases_str)
                for i, basis in enumerate(bases):
                    # the first 4 warps have their own address space
                    if bases_str == "warp_bases" and i < 2:
                        continue
                    if basis == [0, N // 2]:
                        reg_bases[-1], bases[i] = bases[i], reg_bases[-1]
                        return layout_obj
            assert False, f"splitn requires at least one basis of the form [0, N / 2] in lanes or warps[2:] bases. Got {layout}"
    return layout_obj


_compute_tmem_reg_layout.__triton_builtin__ = True


class GluonCallerContext:

    def __init__(self, num_warps: int):
        self.num_warps = num_warps

    def mangle(self):
        return f"_NW{self.num_warps}"

    def initialize_callee(self, fn, builder):
        fn.set_attr("ttg.num-warps", builder.get_int32_attr(self.num_warps))


class GluonSemantic(TritonSemantic[TensorTy]):
    tensor = ttgl.tensor
    lang = ttgl

    builder: GluonOpBuilder

    def __init__(self, builder: GluonOpBuilder):
        self.builder = builder

    def _wrap_handle_infer_layout(self, handle, scalar_ty, shape):
        if shape == []:
            ty = scalar_ty
        else:
            ty = ttgl.distributed_type(scalar_ty, shape, self.builder.get_gluon_layout_from_tensor(handle))
        return self.tensor(handle, ty)

    def _wrap_tensor_infer_layout(self, tensor):
        return self._wrap_handle_infer_layout(tensor.handle, tensor.type.scalar, tensor.shape)

    def _broadcast_shapes(self, lhs_shape: List[int], rhs_shape: List[int]):
        if len(lhs_shape) != len(rhs_shape):
            raise ValueError(f"Cannot broadcast, rank mismatch: {lhs_shape}, {rhs_shape}")

        ret_shape = []
        for i, left in enumerate(lhs_shape):
            right = rhs_shape[i]
            if left == 1:
                ret_shape.append(right)
            elif (right == 1) or (right == left):
                ret_shape.append(left)
            else:
                raise ValueError("Cannot make_shape_compatible: incompatible dimensions "
                                 "at index " + str(i) + ": " + str(left) + " and " + str(right))
        return ret_shape

    def expand_dims(self, input: TensorTy, axis: int) -> TensorTy:
        dst_shape = [ttgl._unwrap_if_constexpr(x) for x in input.shape]
        dst_shape.insert(axis, 1)

        if axis < 0:
            axis += len(input.shape)

        _check(isinstance(input.type, ttgl.distributed_type),
               lambda: f"expected expand_dims input to be a distributed_type but got: {input.type!r}")
        layout = input.type.layout
        _check(isinstance(layout, (SliceLayout, AutoLayout, CoalescedLayout)),
               lambda: f"expected expand_dims input to have a SliceLayout, but got: {layout}")
        _check(
            isinstance(layout, (AutoLayout, CoalescedLayout)) or layout.dim == axis,
            lambda: f"expected expand_dims input layout to be sliced in axis {axis} but got {layout.dim}")

        handle = self.builder.create_expand_dims(input.handle, axis)
        return self._wrap_handle_infer_layout(handle, input.type.scalar, dst_shape)

    def join(self, a: TensorTy, b: TensorTy) -> TensorTy:
        a, b = self.broadcast_impl_value(a, b)
        _check(a.shape != [], lambda: "Cannot join scalars in gluon")
        value = super().join(a, b)
        return self._wrap_tensor_infer_layout(value)

    def split(self, a: TensorTy) -> Tuple[TensorTy, TensorTy]:
        lhs, rhs = super().split(a)
        return self._wrap_tensor_infer_layout(lhs), self._wrap_tensor_infer_layout(rhs)

    def permute(self, input: TensorTy, dims: Tuple[int]) -> TensorTy:
        value = super().permute(input, dims)
        return self._wrap_tensor_infer_layout(value)

    def broadcast_impl_shape(self, input: TensorTy, shape: Tuple[int]) -> TensorTy:
        _check(isinstance(input.type, ttgl.distributed_type),
               lambda: f"expected expand_dims input to be a distributed_type but got: {input.type!r}")
        src_shape = input.type.get_block_shapes()
        _check(len(src_shape) == len(shape), lambda: f"Cannot broadcast, rank mismatch: {src_shape}, {shape}")
        if shape == src_shape:
            return input
        for i, item in enumerate(src_shape):
            if shape[i] != item and item != 1:
                raise ValueError(f"Cannot broadcast, the expanded size of the tensor ({shape[i]})"
                                 f" must match the existing size ({item}) at non-singleton dimension"
                                 f" {i}: {src_shape}, {shape}")
        ret_ty = ttgl.distributed_type(input.type.scalar, shape, input.type.layout)
        handle = self.builder.create_broadcast(input.handle, ret_ty.to_ir(self.builder))
        return self.tensor(handle, ret_ty)

    def broadcast_impl_value(self, lhs: TensorTy, rhs: TensorTy) -> TensorTy:
        lhs_ty = lhs.type
        rhs_ty = rhs.type

        if not lhs_ty.is_block() or not rhs_ty.is_block():
            return super().broadcast_impl_value(lhs, rhs)

        _check(isinstance(lhs_ty, ttgl.distributed_type),
               lambda: f"expected broadcast left input to be a distributed_type but got: {lhs_ty!r}")
        _check(isinstance(rhs_ty, ttgl.distributed_type),
               lambda: f"expected broadcast right input to be a distributed_type but got: {rhs_ty!r}")

        lhs_shape = lhs_ty.get_block_shapes()
        rhs_shape = rhs_ty.get_block_shapes()
        ret_shape = self._broadcast_shapes(lhs_shape, rhs_shape)

        is_lhs_auto = isinstance(lhs_ty.layout, AutoLayout)
        is_rhs_auto = isinstance(rhs_ty.layout, AutoLayout)
        if is_lhs_auto and not is_rhs_auto:
            lhs = self.set_auto_layout(lhs, rhs_ty.layout)
        elif is_rhs_auto and not is_lhs_auto:
            rhs = self.set_auto_layout(rhs, lhs_ty.layout)
        elif lhs_ty.layout != rhs_ty.layout:
            raise ValueError(f"Layout mismatch in broadcast: {lhs_ty.layout} vs {rhs_ty.layout}")

        lhs = self.broadcast_impl_shape(lhs, ret_shape)
        rhs = self.broadcast_impl_shape(rhs, ret_shape)
        return lhs, rhs

    def arange(self, start, end, layout):
        shape = [end - start]
        if layout is None:
            layout = AutoLayout()
        ret_ty = ttgl.distributed_type(ttgl.int32, shape, layout)
        return super().arange(start, end, ret_ty=ret_ty)

    def reshape(self, input: TensorTy, dst_shape: List[int], can_reorder: bool):
        _check(not can_reorder, lambda: "can_reorder is not supported in gluon")
        value = super().reshape(input, dst_shape, can_reorder)
        return self._wrap_tensor_infer_layout(value)

    def splat(self, value, shape, layout):
        if len(shape) == 0:
            return value
        ret_ty = ttgl.distributed_type(value.dtype, shape, layout)
        handle = self.builder.create_splat(ret_ty.to_ir(self.builder), value.handle)
        return ttgl.tensor(handle, ret_ty)

    def full(self, shape, value, dtype, layout):
        scalar = self.make_scalar(value, dtype)
        if layout is None:
            layout = AutoLayout()
        return self.splat(scalar, shape, layout)

    def convert_layout(self, value, layout, assert_trivial=False):
        ty = value.type
        _check(isinstance(ty, ttgl.distributed_type),
               lambda: f"expected convert_layout input to be a distributed_type but got: {ty!r}")
        _check(isinstance(layout, ttgl.DistributedLayout),
               lambda: f"expected 'layout' to be a DistributedLayout but got {layout}")
        ret_ty = ttgl.distributed_type(ty.element_ty, ty.shape, layout)
        ret_ty_ir = ret_ty.to_ir(self.builder)
        if assert_trivial and not self.builder.is_convert_layout_trivial(ret_ty_ir, value.handle):
            raise TypeError(f"layout conversion from {ty.layout} to {layout} is not trivial.\n"
                            f"The linear layouts are:\n{self.to_linear_layout(ty.layout, ty.shape)}\n"
                            f"{self.to_linear_layout(layout, ty.shape)}")
        handle = self.builder.create_convert_layout(ret_ty_ir, value.handle)
        return ttgl.tensor(handle, ret_ty)

    def allocate_shared(self, element_ty, shape, layout, value):
        _check(isinstance(element_ty, ttgl.dtype), lambda: f"expected 'element_ty' to be a dtype but got {element_ty}")
        _check(_is_int_list(shape), lambda: f"all elements of 'shape' must be integers but got {shape}")
        _check(isinstance(layout, ttgl.SharedLayout),
               lambda: f"expected 'layout' to be a SharedLayout but got {layout}")
        ty = ttgl.shared_memory_descriptor_type(element_ty, shape, layout, shape)
        if value is not None:
            handle = self.builder.create_local_alloc(ty.to_ir(self.builder), value.handle)
        else:
            handle = self.builder.create_local_alloc(ty.to_ir(self.builder))
        return ttgl.shared_memory_descriptor(handle, element_ty, shape, layout, shape)

    def shared_load(self, mem_desc, layout):
        _check(isinstance(layout, ttgl.DistributedLayout),
               lambda: f"expected 'layout' to be a DistributedLayout but got {layout}")
        ret_ty = ttgl.distributed_type(mem_desc.dtype, mem_desc.shape, layout)
        handle = self.builder.create_local_load(ret_ty.to_ir(self.builder), mem_desc.handle)
        return ttgl.tensor(handle, ret_ty)

    def shared_store(self, mem_desc, value):
        _check(isinstance(value, ttgl.tensor), lambda: f"expected 'value' to be a tensor, but got a {type(value)}")
        _check(value.shape == mem_desc.shape,
               lambda: f"source shape {value.shape} and destination shape {mem_desc.shape} must match")
        _check(value.dtype == mem_desc.dtype,
               lambda: f"source dtype {value.dtype} and destination dtype {mem_desc.dtype} must match")
        self.builder.create_local_store(mem_desc.handle, value.handle)

    def _check_int_indices_and_normalize_axis(self, mem_desc, indices, axis):
        _check(isinstance(indices, ttgl.tensor),
               lambda: f"expected 'indices' to be a tensor, but got a {type(indices)}")
        _check(isinstance(axis, int), lambda: f"expected 'axis' to be an int, but got a {type(axis)}")
        _check(
            len(indices.shape) == mem_desc.rank,
            lambda: f"indices rank must match memdesc rank: got {len(indices.shape)} and {mem_desc.rank}")
        _check(0 <= axis < mem_desc.rank, lambda: f"axis {axis} is out of bounds for memdesc rank {mem_desc.rank}")
        _check(indices.dtype.is_int(), lambda: f"indices must have integer dtype, got {indices.dtype}")
        return axis

    def _broadcast_shared_scatter_operands(self, mem_desc, values, indices, mask=None):
        _check(isinstance(values, ttgl.tensor), lambda: f"expected 'values' to be a tensor, but got a {type(values)}")
        _check(
            values.dtype == mem_desc.dtype,
            lambda: f"values element type must match destination element type: got {values.dtype} and {mem_desc.dtype}")

        if mask is None:
            values, indices = self.broadcast_tensors(values, indices)
            return values, indices, None

        values, indices, mask = self.broadcast_tensors(values, indices, mask)
        _check(mask.dtype == ttgl.int1, lambda: f"mask must have boolean dtype, got {mask.dtype}")

        return values, indices, mask

    def shared_gather(self, mem_desc, indices, axis):
        axis = self._check_int_indices_and_normalize_axis(mem_desc, indices, axis)

        ret_ty = ttgl.distributed_type(mem_desc.dtype, indices.shape, indices.type.layout)
        handle = self.builder.create_local_gather(ret_ty.to_ir(self.builder), mem_desc.handle, indices.handle, axis)
        return ttgl.tensor(handle, ret_ty)

    def shared_scatter(self, mem_desc, values, indices, axis):
        axis = self._check_int_indices_and_normalize_axis(mem_desc, indices, axis)
        values, indices, _ = self._broadcast_shared_scatter_operands(mem_desc, values, indices)

        self.builder.create_local_scatter(mem_desc.handle, values.handle, indices.handle, axis)

    def _get_shared_atomic_scatter_rmw_op(self, op, dtype):
        if op == "add":
            _check(dtype.is_int() or dtype.is_floating(), lambda: f"atomic_scatter_add does not support dtype {dtype}")
            return ir.ATOMIC_OP.FADD if dtype.is_floating() else ir.ATOMIC_OP.ADD

        if op in ("max", "min"):
            _check(dtype.is_int(), lambda: f"atomic_scatter_{op} does not support dtype {dtype}")
            if dtype.is_int_unsigned():
                return ir.ATOMIC_OP.UMAX if op == "max" else ir.ATOMIC_OP.UMIN
            return ir.ATOMIC_OP.MAX if op == "max" else ir.ATOMIC_OP.MIN

        if op in ("and", "or", "xor"):
            _check(dtype.is_int(), lambda: f"atomic_scatter_{op} does not support dtype {dtype}")
            return {
                "and": ir.ATOMIC_OP.AND,
                "or": ir.ATOMIC_OP.OR,
                "xor": ir.ATOMIC_OP.XOR,
            }[op]

        if op == "xchg":
            _check(dtype.is_int() or dtype.is_floating(), lambda: f"atomic_scatter_xchg does not support dtype {dtype}")
            return ir.ATOMIC_OP.XCHG

        raise ValueError(f"unknown atomic scatter rmw op {op}")

    def shared_atomic_scatter_rmw(self, mem_desc, op, values, indices, axis, mask):
        axis = self._check_int_indices_and_normalize_axis(mem_desc, indices, axis)
        values, indices, mask = self._broadcast_shared_scatter_operands(mem_desc, values, indices, mask)

        mask_handle = mask.handle if mask is not None else None
        rmw_op = self._get_shared_atomic_scatter_rmw_op(op, values.dtype)
        handle = self.builder.create_local_atomic_scatter_rmw(rmw_op, mem_desc.handle, values.handle, indices.handle,
                                                              mask_handle, axis)
        ret_ty = ttgl.distributed_type(mem_desc.dtype, values.shape, values.type.layout)
        return ttgl.tensor(handle, ret_ty)

    def shared_atomic_scatter_add(self, mem_desc, values, indices, axis, mask):
        return self.shared_atomic_scatter_rmw(mem_desc, "add", values, indices, axis, mask)

    def bank_conflicts(self, distr_ty, shared_ty):
        if not isinstance(distr_ty, ttgl.distributed_type):
            raise TypeError(
                f"bank_conflicts expects the register layout to be a distributed_type, got {type(distr_ty)}")

        if not isinstance(shared_ty, ttgl.shared_memory_descriptor_type):
            raise TypeError(
                f"bank_conflicts expects the shared layout to be a shared_memory_descriptor_type, got {type(shared_ty)}"
            )

        if distr_ty.shape != shared_ty.shape:
            raise ValueError(f"register shape {distr_ty.shape} and shared shape {shared_ty.shape} must match")
        if shared_ty.element_ty != distr_ty.element_ty:
            raise ValueError(
                f"mismatched dtypes between register ({distr_ty.element_ty}) and shared ({shared_ty.element_ty}) layouts"
            )
        if shared_ty.shape != shared_ty.alloc_shape[-len(shared_ty.shape):]:
            raise ValueError(
                f"bank_conflicts NYI for subslices. Got shape {shared_ty.shape} and alloc_shape {shared_ty.alloc_shape}"
            )

        reg_attr = distr_ty.layout._to_ir(self.builder)
        shared_attr = shared_ty.layout._to_ir(self.builder)
        return self.builder.get_shared_bank_conflicts(reg_attr, shared_attr, list(distr_ty.shape),
                                                      distr_ty.element_ty.primitive_bitwidth)

    def to_linear_layout(self, layout, shape):
        from triton.experimental.gluon.language.nvidia.blackwell import (
            TensorMemoryLayout,
            TensorMemoryScalesLayout,
        )
        _check(
            isinstance(layout, (DistributedLayout, SharedLayout, TensorMemoryLayout, TensorMemoryScalesLayout)), lambda:
            f"Expected a DistributedLayout, SharedLayout, or TensorMemoryLayout or TensorMemoryScalesLayout, got {type(layout)}"
        )

        if isinstance(layout, (AutoLayout, DistributedLinearLayout, SharedLinearLayout)):
            return ttgl.constexpr(layout)

        return ttgl.constexpr(self.builder.to_linear_layout(layout._to_ir(self.builder), shape))

    def shared_dealloc(self, mem_desc):
        self.builder.create_local_dealloc(mem_desc.handle)

    def set_auto_layout(self, value, layout):
        src_ty = value.type
        _check(isinstance(layout, DistributedLayout),
               lambda: f"set_auto_layout must set to a distributed layout but got {layout}")
        _check(isinstance(src_ty.layout, AutoLayout),
               lambda: f"set_auto_layout input must have auto layout but got {value.type.layout}")
        handle = self.builder.create_set_auto_layout(layout._to_ir(self.builder), value.handle)
        res_ty = ttgl.distributed_type(src_ty.element_ty, src_ty.shape, layout)
        return self.tensor(handle, res_ty)

    def memdesc_slice(self, mem_desc, start, length, dim):
        _check(isinstance(start, int), lambda: f"expected 'start' to be an int but got {start}")
        _check(isinstance(length, int), lambda: f"expected 'length' to be an int but got {length}")
        _check(isinstance(dim, int), lambda: f"expected 'dim' to be an int but got {dim}")
        offsets = [0] * mem_desc.rank
        offsets[dim] = start
        shape = list(mem_desc.shape)
        shape[dim] = length
        layout = mem_desc.layout
        ty = ttgl.shared_memory_descriptor_type(mem_desc.dtype, shape, layout, mem_desc.type.alloc_shape)
        builder = self.builder
        handle = builder.create_memdesc_subslice(ty.to_ir(builder), mem_desc.handle, offsets)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def memdesc_index(self, mem_desc, index):
        index = self.to_tensor(index)
        _check(index.type == ttgl.int32, lambda: f"expected 'index' to be int32 but got {index.type}")
        shape = mem_desc.shape[1:]
        index = self.to_tensor(index).handle
        layout = mem_desc.layout
        ty = ttgl.shared_memory_descriptor_type(mem_desc.dtype, shape, layout, shape)
        builder = self.builder
        handle = builder.create_memdesc_index(ty.to_ir(builder), mem_desc.handle, index)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def memdesc_trans(self, mem_desc, order):
        _check(_is_int_list(order), lambda: f"all elements of 'order' must be integers but got {order}")
        _check(
            len(order) == len(mem_desc.shape),
            lambda: f"source rank ({mem_desc.rank}) and order length ({len(order)}) must match")

        shape = [mem_desc.shape[i] for i in order]
        alloc_shape = mem_desc.type.alloc_shape
        new_alloc_shape = alloc_shape[:len(alloc_shape) - mem_desc.rank]
        new_alloc_shape += [alloc_shape[len(alloc_shape) - mem_desc.rank:][i] for i in order]

        handle = self.builder.create_memdesc_trans(mem_desc.handle, order)
        layout = self.builder.get_gluon_layout_from_memdesc(handle)
        return ttgl.shared_memory_descriptor(handle, element_ty=mem_desc.dtype, shape=shape,
                                             alloc_shape=new_alloc_shape, layout=layout)

    def memdesc_reshape(self, mem_desc, shape):
        _check(_is_int_list(shape), lambda: f"all elements of 'shape' must be integers but got {shape}")
        _check(
            math.prod(shape) == math.prod(mem_desc.shape),
            lambda: (f"memdesc_reshape total elements mismatch: "
                     f"{mem_desc.shape} -> {shape}"),
        )

        handle = self.builder.create_memdesc_reshape(mem_desc.handle, shape)
        layout = self.builder.get_gluon_layout_from_memdesc(handle)
        alloc_shape = mem_desc.type.alloc_shape
        prefix_len = len(alloc_shape) - mem_desc.rank
        new_alloc_shape = alloc_shape[:prefix_len] + list(shape)

        return ttgl.shared_memory_descriptor(
            handle,
            element_ty=mem_desc.dtype,
            shape=shape,
            alloc_shape=new_alloc_shape,
            layout=layout,
        )

    def memdesc_reinterpret(self, mem_desc, dtype, shape, layout):
        _check(isinstance(dtype, ttgl.dtype), lambda: f"expected 'dtype' to be a dtype but got {dtype}")
        _check(_is_int_list(shape), lambda: f"all elements of 'shape' must be integers but got {shape}")
        _check(isinstance(layout, ttgl.SharedLayout),
               lambda: f"expected 'layout' to be a SharedLayout but got {layout}")
        ty = ttgl.shared_memory_descriptor_type(dtype, shape, layout, shape)
        handle = self.builder.create_memdesc_reinterpret(ty.to_ir(self.builder), mem_desc.handle)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def wrap_tensor(self, x, scalar_ty, ret_shape, layout):
        if ret_shape:
            res_ty = ttgl.distributed_type(scalar_ty, ret_shape, layout)
        else:
            res_ty = scalar_ty
        return self.tensor(x, res_ty)

    @staticmethod
    def _check_same_layout(xs):
        for x in xs:
            _check(isinstance(x.type, ttgl.distributed_type), lambda: f"expected distributed_type but got: {x.type!r}")
        layouts = [x.type.layout for x in xs]
        l0 = layouts[0]
        _check(all(l == l0 for l in layouts[1:]),
               lambda: f"Expected inputs to have matching layouts, but got: {layouts}")

    def associative_scan(self, inputs: Sequence[TensorTy], axis: int, region_builder_fn,
                         reverse: bool) -> Tuple[TensorTy, ...]:
        shape = inputs[0].type.shape
        rank = len(shape)

        assert -rank <= axis < rank, f"scan axis {axis} must be < inputs rank ({rank})"

        if axis < 0:
            axis += rank

        for t in inputs:
            assert t.type.shape == shape, "all scan inputs must have the same shape"

        scan_op = self.builder.create_scan([t.handle for t in inputs], axis, reverse)
        region_builder_fn(scan_op)
        assert scan_op.verify()

        return tuple(
            self._wrap_handle_infer_layout(scan_op.get_result(i), inputs[i].type.scalar, shape)
            for i in range(len(inputs)))

    def reduction(self, inputs: Sequence[TensorTy], axis: int, region_builder_fn) -> Tuple[TensorTy, ...]:
        if axis is None:
            inputs = tuple(self.reshape(t, [t.numel.value], can_reorder=False) for t in inputs)
            axis = 0
        # get result shape
        shape = inputs[0].type.shape
        rank = len(shape)
        _check(0 <= axis < rank, lambda: f"expected reduction axis to be in the range [0, {rank}) but got {axis}")
        self._check_same_layout(inputs)
        ret_shape = [s for i, s in enumerate(shape) if i != axis]
        assert all(t.type.shape == shape for t in inputs), "all reduction inputs must have the same shape"

        reduce_op = self.builder.create_reduce([t.handle for t in inputs], axis)
        region_builder_fn(reduce_op)
        assert reduce_op.verify()

        return tuple(
            self._wrap_handle_infer_layout(reduce_op.get_result(i), inputs[i].type.scalar, ret_shape)
            for i in range(len(inputs)))

    def histogram(self, input: TensorTy, num_bins: int, mask: TensorTy, layout) -> TensorTy:
        _check(len(input.shape) == 1, lambda: "histogram only supports 1D input")
        _check(input.dtype.is_int(), lambda: "histogram only supports integer input")
        _check(layout is not None, lambda: "histogram requires a destination layout")
        if mask is not None:
            mask, input = self.broadcast_impl_value(mask, input)
            _check(mask.type.scalar.is_bool(), lambda: "Mask must have boolean scalar type")
            mask = mask.handle
        layout_attr = layout._to_ir(self.builder)
        handle = self.builder.create_histogram(input.handle, num_bins, mask, layout_attr)
        return self.wrap_tensor(handle, ttgl.int32, [num_bins], layout)

    def cat(self, lhs: TensorTy, rhs: TensorTy, can_reorder: bool, layout) -> TensorTy:
        _check(layout is not None, lambda: "cat requires a destination layout")
        _check(can_reorder, lambda: "current implementation of `cat` always may reorder elements")
        _check(len(lhs.shape) == 1, lambda: "cat requires a rank-1 input")
        ret_type = ttgl.distributed_type(lhs.type.scalar, [lhs.shape[0] + rhs.shape[0]], layout)
        return self.tensor(self.builder.create_cat(lhs.handle, rhs.handle, ret_type.to_ir(self.builder)), ret_type)

    def gather(self, src: TensorTy, index: TensorTy, axis: int) -> TensorTy:
        _check(isinstance(src.type, ttgl.distributed_type), lambda: f"expected distributed_type but got: {src.type!r}")
        _check(isinstance(index.type, ttgl.distributed_type),
               lambda: f"expected distributed_type but got: {index.type!r}")
        _check(index.type.scalar.is_int(), lambda: f"expected integer scalar type but got: {index.type.scalar!r}")

        rank = len(src.type.shape)
        _check(len(index.type.shape) == rank, lambda: "source and index tensors must have the same rank")
        _check(-rank <= axis < rank, lambda: f"gather axis {axis} must be < source rank ({rank})")
        if axis < 0:
            axis += rank

        for d in range(rank):
            if d == axis:
                continue
            _check(
                index.type.shape[d] == src.type.shape[d],
                lambda: f"index dim {axis} must match the corresponding source dim",
            )
        gather = self.builder.create_gather(src.handle, index.handle, axis)
        return self.wrap_tensor(gather, src.type.scalar, index.type.shape, index.type.layout)

    def fp4_to_fp(self, src: TensorTy, elem_type, axis) -> TensorTy:
        result = self.builder.create_fp4_to_fp(src.handle, elem_type.to_ir(self.builder), axis)
        shape = list(src.type.shape)
        shape[axis] *= 2
        return self._wrap_handle_infer_layout(result, elem_type, shape)

    def warp_specialize(self, functions_and_args, worker_num_warps: Sequence[int], worker_num_regs: Sequence[int],
                        generator):
        for _, args in functions_and_args:
            _check(isinstance(args, (tuple, ttgl.tuple)),
                   lambda: f"function arguments must be a tuple of arguments, but got {type(args)}")

        assert len(functions_and_args) >= 1, "expected at least one function for the default partition"
        default_partition, default_args = functions_and_args[0]
        num_partitions = len(functions_and_args) - 1
        workers = functions_and_args[1:]

        assert num_partitions == len(
            worker_num_warps
        ), f"warp specialize got {num_partitions} partitions but {len(worker_num_warps)} warp counts"

        if worker_num_regs is not None:
            assert num_partitions == len(
                worker_num_regs
            ), f"warp specialize got {num_partitions} partitions but {len(worker_num_regs)} register counts"

        builder = self.builder
        insert_pt = builder.get_insertion_point()

        # Emit the default partition to get the result types.
        default_block = builder.new_block()
        builder.set_insertion_point_to_start(default_block)
        default_result = generator.call_JitFunction(default_partition, default_args, kwargs={})
        mlir_results = flatten_values_to_ir([default_result])
        builder.create_warp_yield(mlir_results)
        result_types = [r.get_type() for r in mlir_results]

        # Create the warp specialize op.
        worker_args = [flatten_values_to_ir(args) for _, args in workers]
        mlir_args = sum(worker_args, [])
        builder.restore_insertion_point(insert_pt)
        ws_op = builder.create_warp_specialize(result_types, worker_num_warps)
        ws_op.get_default_region().push_back(default_block)

        if worker_num_regs is not None:
            ws_op.set_requested_registers(worker_num_regs)

        # Emit the partition regions.
        builder.create_block_with_parent(ws_op.get_partition_op_holder(), [])
        partitions_op = builder.create_warp_specialize_partitions(mlir_args, num_partitions)
        arg_types = [arg.get_type() for arg in mlir_args]
        arg_it = 0
        for i, (func, args) in enumerate(workers):
            caller_context = GluonCallerContext(num_warps=worker_num_warps[i])
            block = builder.create_block_with_parent(partitions_op.get_region(i), arg_types)
            mlir_args = worker_args[i]
            block_args = [block.get_argument(arg_it + j) for j in range(len(mlir_args))]
            block_args = unflatten_ir_values(block_args, [arg.type for arg in args])
            generator.call_JitFunction(func, block_args, kwargs={}, caller_context=caller_context)
            builder.create_warp_return()
            arg_it += len(mlir_args)

        builder.set_insertion_point_after(ws_op.get_operation())
        mlir_results = [ws_op.get_result(i) for i in range(len(result_types))]
        return next(unflatten_ir_values(mlir_results, [default_result.type]))

    def num_ctas(self):
        return ttgl.constexpr(self.builder.options.num_ctas)

    def num_warps(self, generator):
        if generator.caller_context is not None:
            assert isinstance(generator.caller_context, GluonCallerContext)
            return ttgl.constexpr(generator.caller_context.num_warps)
        return ttgl.constexpr(self.builder.options.num_warps)
<|endoftext|>from dataclasses import dataclass, field
import itertools
import math
from typing import List

from triton.language.core import _unwrap_if_constexpr, _unwrap_shape, constexpr_type
from triton.runtime.jit import constexpr_function
from triton._C.libtriton import gluon_ir


class DistributedLayout:
    """
    Base class for distributed memory layouts in Gluon IR.
    """

    @property
    def type(self):
        return constexpr_type(self)

    @property
    def rank(self):
        raise NotImplementedError("DistributedLayout subclasses must define rank")

    def format_tensor_view(self, shape: list[int]) -> str:
        return gluon_ir.get_layout_view(self, [_unwrap_if_constexpr(s) for s in shape], False)

    def format_hardware_view(self, shape: list[int]) -> str:
        return gluon_ir.get_layout_view(self, [_unwrap_if_constexpr(s) for s in shape], True)


@dataclass(frozen=True)
class AutoLayout(DistributedLayout):

    def _to_ir(self, builder):
        return builder.get_auto_layout()

    def mangle(self):
        return "AL"

    @property
    def rank(self):
        raise ValueError("AutoLayout has no rank")


@dataclass(frozen=True)
class CoalescedLayout(DistributedLayout):

    def _to_ir(self, builder):
        return builder.get_coalesced_layout()

    def mangle(self):
        return "CL"

    @property
    def rank(self):
        raise ValueError("CoalescedLayout has no rank")


@dataclass(frozen=True)
class BlockedLayout(DistributedLayout):
    """
    Represents a blocked layout, partitioning a tensor across threads, warps, and CTAs.

    Args:
        size_per_thread (List[int]): Number of elements per thread per dimension.
        threads_per_warp (List[int]): Number of threads per warp per dimension.
        warps_per_cta (List[int]): Number of warps per CTA per dimension.
        order (List[int]): The ordering of dimensions for partitioning.
        cga_layout (Optional[List[List[int]]]): Bases describing how CTAs tile each dimension.
    """
    size_per_thread: List[int]
    threads_per_warp: List[int]
    warps_per_cta: List[int]
    order: List[int]
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("size_per_thread", _unwrap_if_constexpr(self.size_per_thread))
        super().__setattr__("threads_per_warp", _unwrap_if_constexpr(self.threads_per_warp))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("order", _unwrap_if_constexpr(self.order))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))

        rank = len(self.size_per_thread)
        assert len(self.threads_per_warp) == rank
        assert len(self.warps_per_cta) == rank
        assert len(self.order) == rank

    def _to_ir(self, builder):
        return builder.get_blocked_layout(
            self.size_per_thread,
            self.threads_per_warp,
            self.warps_per_cta,
            self.order,
            self.cga_layout,
        )

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        size_per_thread = stringify(self.size_per_thread)
        threads_per_warp = stringify(self.threads_per_warp)
        warps_per_cta = stringify(self.warps_per_cta)
        order = stringify(self.order)
        cga_layout = "_".join("~".join(map(str, vec)) for vec in self.cga_layout) if self.cga_layout else ""
        return f"B{size_per_thread}_{threads_per_warp}_{warps_per_cta}_{order}_{cga_layout}B"

    def __hash__(self):
        return hash((tuple(self.size_per_thread), tuple(self.threads_per_warp), tuple(self.warps_per_cta),
                     tuple(self.order), tuple(tuple(vec) for vec in self.cga_layout)))

    @property
    def rank(self):
        return len(self.order)


@dataclass(frozen=True)
class SliceLayout(DistributedLayout):
    """
    Represents a layout corresponding to slicing a distributed tensor along one dimension.

    Args:
        dim (int): The dimension index to slice.
        parent (DistributedLayout): The parent layout before slicing.
    """
    dim: int
    parent: DistributedLayout

    def __post_init__(self):
        super().__setattr__("dim", _unwrap_if_constexpr(self.dim))
        super().__setattr__("parent", _unwrap_if_constexpr(self.parent))

    def _to_ir(self, builder):
        return builder.get_slice_layout(
            self.dim,
            self.parent._to_ir(builder),
        )

    def mangle(self) -> str:
        return f"SL{self.dim}_{self.parent.mangle()}SL"

    def __hash__(self):
        return hash((self.dim, self.parent))

    @property
    def rank(self):
        return self.parent.rank - 1

    @property
    def cga_layout(self):
        parent_cga_layout = self.parent.cga_layout
        if not parent_cga_layout:
            return []

        rank = self.parent.rank
        assert 0 <= self.dim < rank
        return [basis[:self.dim] + basis[self.dim + 1:] for basis in parent_cga_layout]


@dataclass(frozen=True)
class DistributedLinearLayout(DistributedLayout):
    """
    Represents a linear distributed layout with explicit bases at register, lane, warp, and block levels.
    See: https://arxiv.org/abs/2505.23819 for reference.

    Args:
        reg_bases (List[List[int]]): Bases for register-level distribution.
        lane_bases (List[List[int]]): Bases for lane-level distribution.
        warp_bases (List[List[int]]): Bases for warp-level distribution.
        block_bases (List[List[int]]): Bases for block-level distribution.
        shape (List[int]): The tensor global shape.
    """
    reg_bases: List[List[int]]
    lane_bases: List[List[int]]
    warp_bases: List[List[int]]
    block_bases: List[List[int]]
    shape: List[int]

    def __post_init__(self):
        super().__setattr__("reg_bases", _unwrap_shape(self.reg_bases))
        super().__setattr__("lane_bases", _unwrap_shape(self.lane_bases))
        super().__setattr__("warp_bases", _unwrap_shape(self.warp_bases))
        super().__setattr__("block_bases", _unwrap_shape(self.block_bases))
        super().__setattr__("shape", _unwrap_shape(self.shape))

        rank = len(self.shape)

        for basis in self.reg_bases:
            assert len(basis) == rank
        for basis in self.lane_bases:
            assert len(basis) == rank
        for basis in self.warp_bases:
            assert len(basis) == rank
        for basis in self.block_bases:
            assert len(basis) == rank

    def _to_ir(self, builder):
        return builder.get_distributed_linear_layout(self.reg_bases, self.lane_bases, self.warp_bases, self.block_bases,
                                                     self.shape)

    def mangle(self):
        return f"DLL{self.reg_bases}_{self.lane_bases}_{self.warp_bases}_{self.block_bases}_{self.shape}DLL"

    def __hash__(self):
        return hash((
            tuple(map(tuple, self.reg_bases)),
            tuple(map(tuple, self.lane_bases)),
            tuple(map(tuple, self.warp_bases)),
            tuple(map(tuple, self.block_bases)),
            tuple(self.shape),
        ))

    @property
    def rank(self):
        return len(self.shape)


@dataclass(frozen=True)
class DotOperandLayout(DistributedLayout):
    """
    Represents a layout for a dot operand.

    Args:
        operand_index (int): 0 for LHS and 1 for RHS of the dot operation.
        parent (DistributedLayout): The parent layout, representing the MMA.
        k_width (int): Number of elements per 32-bits.
    """
    operand_index: int
    parent: DistributedLayout
    k_width: int

    def __post_init__(self):
        super().__setattr__("operand_index", _unwrap_if_constexpr(self.operand_index))
        super().__setattr__("parent", _unwrap_if_constexpr(self.parent))
        super().__setattr__("k_width", _unwrap_if_constexpr(self.k_width))

    def _to_ir(self, builder):
        return builder.get_dot_operand_layout(self.operand_index, self.parent._to_ir(builder), self.k_width)

    def mangle(self) -> str:
        return f"DO{self.operand_index}_{self.parent.mangle()}_{self.k_width}DO"

    def __hash__(self):
        return hash((self.operand_index, self.parent, self.k_width))

    @property
    def rank(self):
        return self.parent.rank

    @property
    def cga_layout(self):
        parent_cga_layout = _unwrap_if_constexpr(getattr(self.parent, "cga_layout", [])) or []
        if not parent_cga_layout:
            return []

        rank = self.parent.rank
        assert all(len(basis) == rank for basis in parent_cga_layout)

        k_dim = rank - 1 if self.operand_index == 0 else rank - 2
        assert 0 <= k_dim < rank

        derived = []
        for basis in parent_cga_layout:
            new_basis = list(basis)
            new_basis[k_dim] = 0
            derived.append(new_basis)
        return derived


@dataclass(frozen=True, eq=True)
class NVMMADistributedLayout(DistributedLayout):
    """
    Represents a layout for NVIDIA MMA (tensor core) operations.

    Args:
        version (List[int]): Version identifier for the MMA instruction.
        warps_per_cta (List[int]): Number of warps per CTA.
        instr_shape (List[int]): Instruction shape for MMA.
        cga_layout (Optional[List[List[int]]]): Bases describing CTA tiling.
    """
    version: List[int]
    warps_per_cta: List[int]
    instr_shape: List[int]
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("version", _unwrap_if_constexpr(self.version))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("instr_shape", _unwrap_if_constexpr(self.instr_shape))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))

    def _to_ir(self, builder):
        return builder.get_mma_layout(
            self.version,
            self.warps_per_cta,
            self.cga_layout,
            self.instr_shape,
        )

    def mangle(self) -> str:
        cga_layout = "_".join("~".join(map(str, vec)) for vec in self.cga_layout) if self.cga_layout else ""
        return f"MMA_{self.version}_{self.warps_per_cta}_{self.instr_shape}_{cga_layout}_MMA"

    def __hash__(self):
        return hash((tuple(self.version), tuple(self.warps_per_cta), tuple(self.instr_shape),
                     tuple(tuple(vec) for vec in self.cga_layout)))

    @property
    def rank(self):
        return len(self.warps_per_cta)


class SharedLayout:
    """
    Base class for shared memory layouts in Gluon IR.
    """

    @property
    def type(self):
        return constexpr_type(self)

    def format_tensor_view(self, shape: list[int]) -> str:
        return gluon_ir.get_layout_view(self, [_unwrap_if_constexpr(s) for s in shape], False)

    def format_hardware_view(self, shape: list[int]) -> str:
        return gluon_ir.get_layout_view(self, [_unwrap_if_constexpr(s) for s in shape], True)


@constexpr_function
def _get_shape_per_cta(shape, cga_layout):
    if not cga_layout:
        return shape
    shape_per_cta = list(shape)
    rank = len(cga_layout[0])
    cga_shape = [0] * rank
    for basis in cga_layout:
        assert len(basis) == rank
        for i in range(rank):
            cga_shape[i] = max(cga_shape[i], basis[i])
    # The shape is the largest stride * 2, or 1 if the stride was always zero
    for i in range(rank):
        if cga_shape[i] == 0:
            cga_shape[i] = 1
        else:
            cga_shape[i] *= 2
    for dim in range(rank):
        assert shape_per_cta[dim] % cga_shape[dim] == 0, f"Shape {shape} is not divisible by CGA layout {cga_layout}"
        shape_per_cta[dim] //= cga_shape[dim]
    return shape_per_cta


@dataclass(frozen=True)
class NVMMASharedLayout(SharedLayout):
    """
    Represents a layout for shared memory suitable for NVIDIA MMA operations.

    Args:
        swizzle_byte_width (int): Width in bytes for swizzling.
        element_bitwidth (int): Bitwidth of element type.
        rank (int): Rank of the tensor.
        transposed (bool): Whether the layout is transposed.
        fp4_padded (bool): Whether FP4 padding is used.
        cga_layout (Optional[List[List[int]]]): Bases describing CTA tiling.
    """
    swizzle_byte_width: int
    element_bitwidth: int
    rank: int = 2
    transposed: bool = False
    fp4_padded: bool = False
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("swizzle_byte_width", _unwrap_if_constexpr(self.swizzle_byte_width))
        super().__setattr__("element_bitwidth", _unwrap_if_constexpr(self.element_bitwidth))
        super().__setattr__("transposed", _unwrap_if_constexpr(self.transposed))
        super().__setattr__("fp4_padded", _unwrap_if_constexpr(self.fp4_padded))

        # TODO: Make rank optional and check that (rank or cga_layout)
        cga_layout = self.cga_layout or []
        if cga_layout:
            assert len(cga_layout[0]) == self.rank

        super().__setattr__("rank", _unwrap_if_constexpr(self.rank))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(cga_layout))

        assert self.element_bitwidth in [8, 16, 32, 64]
        assert self.swizzle_byte_width in [0, 32, 64, 128]

        if self.fp4_padded:
            assert self.swizzle_byte_width == 128, "fp4_padded only supports 128 byte swizzling"
            assert self.element_bitwidth == 8, "fp4_padded is only supported for element_bitwidth=8"

    def _to_ir(self, builder):
        return builder.get_nvmma_shared_layout(
            self.swizzle_byte_width,
            self.element_bitwidth,
            self.transposed,
            self.fp4_padded,
            self.cga_layout,
            self.rank,
        )

    @staticmethod
    @constexpr_function
    def get_default_for(block_shape, dtype, transposed=False, fp4_padded=False, cga_layout=None):
        """Returns an NVMMASharedLayout with default swizzling for a given shape.

        This picks the largest swizzle pattern compatible with the shape, which
        allows emitting the fewest TMA or MMA messages.
        """
        packing_factor = 2 if fp4_padded else 1
        shape_per_cta = block_shape if cga_layout is None else _get_shape_per_cta(block_shape, cga_layout)
        rank = len(block_shape)
        if transposed:
            shape_per_cta = shape_per_cta[1:] + shape_per_cta[:1]
        contig_dim_size = shape_per_cta[-1] * packing_factor
        contig_dim_bytes = contig_dim_size * dtype.primitive_bitwidth // 8
        if contig_dim_bytes >= 128 and contig_dim_bytes % 128 == 0:
            swizzle_byte_width = 128
        elif contig_dim_bytes >= 64 and contig_dim_bytes % 64 == 0:
            swizzle_byte_width = 64
        elif contig_dim_bytes >= 32 and contig_dim_bytes % 32 == 0:
            swizzle_byte_width = 32
        else:
            swizzle_byte_width = 0

        flatten_outer_dim = 1
        for size in shape_per_cta[:-1]:
            flatten_outer_dim *= size
        if len(block_shape) < 2 or flatten_outer_dim < 8:
            swizzle_byte_width = 0

        return NVMMASharedLayout(
            swizzle_byte_width=swizzle_byte_width,
            element_bitwidth=dtype.primitive_bitwidth,
            rank=rank,
            transposed=transposed,
            fp4_padded=fp4_padded,
            cga_layout=cga_layout,
        )

    def mangle(self) -> str:
        cga_layout = "_".join("~".join(map(str, vec)) for vec in self.cga_layout) if self.cga_layout else ""
        return f"NVMMA_{self.swizzle_byte_width}_{self.element_bitwidth}_{self.transposed}_{self.fp4_padded}_{cga_layout}_NVMMA"

    def __hash__(self):
        return hash((self.swizzle_byte_width, self.element_bitwidth, self.rank, self.transposed, self.fp4_padded,
                     tuple(tuple(vec) for vec in self.cga_layout) if self.cga_layout else None))


@dataclass(frozen=True, eq=True)
class SwizzledSharedLayout(SharedLayout):
    """
    Represents a generic swizzled shared memory layout.

    Args:
        vec (int): Vector width for swizzling.
        per_phase (int): Elements per swizzle phase.
        max_phase (int): Maximum number of swizzle phases.
        order (List[int]): Dimension ordering for swizzling.
        cga_layout (Optional[List[List[int]]]): Bases describing CTA tiling.
    """
    vec: int
    per_phase: int
    max_phase: int
    order: List[int]
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("vec", _unwrap_if_constexpr(self.vec))
        super().__setattr__("per_phase", _unwrap_if_constexpr(self.per_phase))
        super().__setattr__("max_phase", _unwrap_if_constexpr(self.max_phase))
        super().__setattr__("order", _unwrap_if_constexpr(self.order))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))

    def _to_ir(self, builder):
        return builder.get_swizzled_shared_layout(
            self.vec,
            self.per_phase,
            self.max_phase,
            self.order,
            self.cga_layout,
        )

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        cga_layout = "_".join("~".join(map(str, vec)) for vec in self.cga_layout) if self.cga_layout else ""
        return f"SSS_{self.vec}_{self.per_phase}_{self.max_phase}_{stringify(self.order)}_{cga_layout}_SSS"

    def __hash__(self):
        return hash(
            (self.vec, self.per_phase, self.max_phase, tuple(self.order), tuple(tuple(vec) for vec in self.cga_layout)))


@dataclass(frozen=True, eq=True)
class PaddedSharedLayout(SharedLayout):
    """
    Represents a layout for the access to shared memory. Compared to SwizzledSharedLayout,
    it combined padding and element reordering via linear transformation (e.g. row permutation)
    to avoid shared memory bank conflicts. After every interval tensor elements, the
    corresponding number of padding elements are inserted. If a position corresponds to
    multiple intervals, the padding amounts are summed.

    In the following example of a tensor,
    `eM` represents original elements in the and `pN` represents padded element.

    Before padding, the shared memory looks like:
    [e0, e1,
     e2, e3,
     e4, e5,
     e6, e7,
     ...]

    After padding with interval-padding list [[2, 1], [4, 2]] with an identity remapping,
    the shared memory will be
    [e0, e1, p0,
     e2, e3, p1, p2, p3,
     e4, e5, p4,
     e6, e7, p5, p6, p7,
     ...]

    Furthermore this encoding allows for a linear remapping from the 1-D shared
    memory offset to logical n-D tensor elements. The remapping is given in the form
    of linear bases mapping from offset to [dim0, dim1...dimN-1].
    See LinearLayout.h for more details how linear layouts are applied to remap
    elements.
    Some concrete examples using `xN` and `yN` to mean the logical n-D tensor elements
    and `pN` to mean padding:

    After padding for shape = [8] with interval-padding list [[2, 2]], offset_bases = [[2], [1]] and cga_layout = []:
    [x0, x2, p0 p1, x1, x3]

    After padding for shape = [8, 4] with interval_padding_pairs = [[8, 1]], offset_bases = [[0, 1], [0, 2], /*gap, stride by 2 rows*/[2, 0], [4, 0], [1, 0]]] and cga_layout = []:
    [
        x0y0, x0y1, x0y2, x0y3,
        x2y0, x2y1, x2y2, x2y3,
        p0,
        x4y0, x4y1, x4y2, x4y3,
        x6y0, x6y1, x6y2, x6y3,
        p1,
        x1y0, x1y1, x1y2, x1y3,
        x3y0, x3y1, x3y2, x3y3,
        p2,
        x5y0, x5y1, x5y2, x5y3,
        x7y0, x7y1, x7y2, x7y3,
    ]

    Args:
        interval_padding_pairs (List[int]): List of [interval, padding] pair and both interval and padding must be powers of 2.
        offset_bases (List[int]): Bases for shared memory offsets
        cga_layout (List[List[int]]): Bases for block-level shared memory offsets.
        shape (List[int]): n-D logical shared memory shape
    """
    interval_padding_pairs: List[List[int]]
    offset_bases: List[List[int]]
    cga_layout: List[List[int]]
    shape: List[int]

    def __post_init__(self):
        super().__setattr__("interval_padding_pairs", _unwrap_shape(self.interval_padding_pairs))
        super().__setattr__("offset_bases", _unwrap_shape(self.offset_bases))
        super().__setattr__("cga_layout", _unwrap_shape(self.cga_layout))
        super().__setattr__("shape", _unwrap_shape(self.shape))

        rank = len(self.shape)

        for basis in self.offset_bases:
            assert len(basis) == rank
        for basis in self.cga_layout:
            assert len(basis) == rank

        self.verify()

    def _to_ir(self, builder):
        intervals, paddings = zip(*self.interval_padding_pairs)
        return builder.get_padded_shared_layout(intervals, paddings, self.offset_bases, self.cga_layout, self.shape)

    def mangle(self) -> str:
        return f"PaddedShared_{self.interval_padding_pairs}_{self.offset_bases}_{self.cga_layout}_{self.shape}_PaddedShared"

    def verify(self):
        pairs = self.interval_padding_pairs
        assert len(pairs) > 0, "PaddedSharedLayout interval_padding_pairs must have at least one interval-padding pair"
        assert all(len(pair) == 2 for pair in pairs)
        intervals, paddings = zip(*pairs)

        unique_intervals = list(set(intervals))
        assert len(unique_intervals) == len(intervals)

        is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
        assert all(is_power_of_2(n) for n in intervals), "PaddedSharedLayout interval values must all be power of two"
        assert all(is_power_of_2(n) for n in paddings), "PaddedSharedLayout padding values must all be power of two"

        rank = len(self.shape)
        assert rank > 0, "PaddedSharedLayout order must not be empty"

    @staticmethod
    @constexpr_function
    def with_identity_for(interval_padding_pairs, shape, order, cga_layout=[]):
        """Returns a PaddedSharedLayout with the given interval and padding pairs and an identity mapping as the linear component for the given shape and order.
        """
        assert len(shape) == len(order)
        is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
        assert all(is_power_of_2(n) for n in shape)

        rank = len(shape)
        shape_per_cta = _get_shape_per_cta(shape, cga_layout) if cga_layout else shape
        # Create a idendity mapping based on shape_per_cta + order
        offset_bases = []
        for dim in order:
            for basis in range(int(math.log2(shape_per_cta[dim]))):
                offset_bases.append([1 << basis if i == dim else 0 for i in range(rank)])

        return PaddedSharedLayout(interval_padding_pairs, offset_bases, cga_layout, shape)

    def __hash__(self):
        return hash((tuple(map(tuple, self.interval_padding_pairs)), tuple(map(tuple, self.offset_bases)),
                     tuple(map(tuple, self.cga_layout)), tuple(self.shape)))


@dataclass(frozen=True)
class SharedLinearLayout(SharedLayout):
    """Represents a shared memory layout defined via an explicit LinearLayout."""

    offset_bases: List[List[int]]
    block_bases: List[List[int]] = field(default_factory=list)
    alignment: int = 16

    def __post_init__(self):
        super().__setattr__("offset_bases", _unwrap_shape(self.offset_bases))
        super().__setattr__("block_bases", _unwrap_shape(self.block_bases))
        super().__setattr__("alignment", _unwrap_if_constexpr(self.alignment))

        assert len(self.offset_bases) != 0, "SharedLinearLayout offset_bases must not be empty"
        rank = len(self.offset_bases[0])
        assert rank > 0, "SharedLinearLayout offset_bases must not be empty"
        for basis in self.offset_bases:
            assert len(basis) == rank
        for basis in self.block_bases:
            assert len(basis) == rank
        assert self.alignment > 0 and (self.alignment & (self.alignment - 1)) == 0, \
            "SharedLinearLayout alignment must be a positive power of two"

    def _to_ir(self, builder):
        return builder.get_shared_linear_layout(self.offset_bases, self.block_bases, self.alignment)

    def mangle(self) -> str:
        return f"SharedLinear_{self.offset_bases}_{self.block_bases}_{self.alignment}_SharedLinear"

    @property
    def shape(self):
        rank = len(self.offset_bases[0])
        max_stride = [0] * rank
        for b in itertools.chain(self.offset_bases, self.block_bases):
            for i, bi in enumerate(b):
                max_stride[i] = max(max_stride[i], bi)
        return [1 << s.bit_length() for s in max_stride]

    def __hash__(self):
        return hash((
            tuple(map(tuple, self.offset_bases)),
            tuple(map(tuple, self.block_bases)),
            self.alignment,
        ))


# Python impl of LinearEncodingAttr::basesPerDim
def bases_per_dim(bases, rank, skip_broadcast=True):
    result = [1] * rank

    if not bases:
        return result

    non_zero_idx = None

    for basis in bases:
        # Find the first non-zero index in the current basis
        idx = next((i for i, v in enumerate(basis) if v != 0), None)
        if idx is not None:
            non_zero_idx = idx
            result[idx] *= 2
        elif not skip_broadcast:
            # If no non-zero found and we're not skipping broadcasts, use the last found non-zero index
            assert non_zero_idx is not None
            result[non_zero_idx] *= 2

    return result


def warps_per_cta(layout, shape):
    if hasattr(layout, 'warp_bases'):
        return bases_per_dim(layout.warp_bases, len(shape))
    elif isinstance(layout, (SliceLayout, DotOperandLayout)):
        return warps_per_cta(layout.parent, shape)
    else:
        return layout.warps_per_cta
<|endoftext|>from typing import TypeVar
from triton.runtime.jit import JITFunction
import triton.language.standard as tl_standard
from .._runtime import GluonJITFunction, jit
from triton import knobs
from . import _core as ttgl

T = TypeVar("T")


def _import_from_triton(fn: JITFunction[T]) -> GluonJITFunction[T]:
    assert knobs.runtime.interpret or isinstance(fn, JITFunction)
    # Wrap the function and preserve its original docstring
    gluon_fn = jit(fn.fn)
    gluon_fn.__doc__ = fn.__doc__
    return gluon_fn


cdiv = _import_from_triton(tl_standard.cdiv)
sum = _import_from_triton(tl_standard.sum)
max = _import_from_triton(tl_standard.max)
min = _import_from_triton(tl_standard.min)
ravel = _import_from_triton(tl_standard.ravel)
reduce_or = _import_from_triton(tl_standard.reduce_or)
xor_sum = _import_from_triton(tl_standard.xor_sum)


@jit
def zeros(shape, dtype, layout=None):
    """
    Create a tensor filled with zeros.

    Args:
        shape (Sequence[int]): The shape of the tensor.
        dtype (dtype): The data type for the tensor.
        layout (Optional[DistributedLayout]): The distributed layout of the tensor, defaults to AutoLayout().

    Returns:
        tensor: A tensor where every element is zero.
    """
    return ttgl.full(shape, 0, dtype, layout)


@jit
def full_like(input, value, shape=None, dtype=None, layout=None):
    """
    Create a tensor with the same properties as a given tensor, filled with a specified value.

    Args:
        input (tensor): Reference tensor to infer default shape, dtype, and layout.
        value (int or float): The fill value.
        shape (Sequence[int], optional): Target shape. Defaults to input.shape.
        dtype (dtype, optional): Target data type. Defaults to input.dtype.
        layout (DistributedLayout, optional): Target layout. Defaults to input.layout.

    Returns:
        tensor: A tensor where every element equals value.
    """
    return ttgl.full(
        input.shape if shape is None else shape,
        value,
        input.dtype if dtype is None else dtype,
        input.type.layout if layout is None else layout,
    )


@jit
def zeros_like(input, shape=None, dtype=None, layout=None):
    """
    Create a tensor with the same properties as a given tensor, filled with zeros.

    Args:
        input (tensor): Reference tensor to infer default shape, dtype, and layout.
        shape (Sequence[int], optional): Target shape. Defaults to input.shape.
        dtype (dtype, optional): Target data type. Defaults to input.dtype.
        layout (DistributedLayout, optional): Target layout. Defaults to input.layout.

    Returns:
        tensor: A tensor where every element is zero.
    """
    return full_like(input, 0, shape=shape, dtype=dtype, layout=layout)
<|endoftext|>from __future__ import annotations
import inspect
import math
from typing import Callable, TypeVar, List, TYPE_CHECKING, Tuple
from functools import wraps
import warnings

if TYPE_CHECKING:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
    from ._semantic import GluonSemantic

from ._layouts import (SharedLayout, DistributedLayout, BlockedLayout, DotOperandLayout, AutoLayout, CoalescedLayout,
                       SharedLinearLayout, _get_shape_per_cta)
from triton._C.libtriton import ir
import triton.language.core as tl_core
from triton.language.core import (
    aggregate_replace,
    base_value,
    base_type,
    constexpr,
    dtype,
    block_type,  # TODO: block type with layout info
    pointer_type,
    void,
    int1,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float8e5,
    float8e5b16,
    float8e4nv,
    float8e4b8,
    float8e4b15,
    float16,
    bfloat16,
    float32,
    float64,
    _unwrap_if_constexpr,
    _unwrap_shape,
    static_range,
    tensor,
    tuple,
    tuple_type,
)

# We define __all__ only to appease the python linter, these are not used in
# this file but we want to import them anyway so they are importable from here.
__all__ = [
    "aggregate_replace",
    "constexpr",
    "pointer_type",
    "void",
    "int1",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float8e5",
    "float8e5b16",
    "float8e4nv",
    "float8e4b8",
    "float8e4b15",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "distributed_type",
    "shared_memory_descriptor_type",
    "static_range",
    "tuple",
    "tuple_type",
    "num_ctas",
]

T = TypeVar("T")

# TODO: split these
GLUON_BUILTIN = "__triton_builtin__"


def builtin(fn: T) -> T:
    """Mark a function as a builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_semantic" not in kwargs or kwargs["_semantic"] is None:
            raise ValueError("Did you forget to add @triton.gluon.jit ? "
                             "(`_semantic` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, GLUON_BUILTIN, True)
    wrapper.signature = inspect.signature(fn)

    return wrapper


# Explicitly import forwarded Triton language symbols so mypy sees them.
add = builtin(tl_core.add)
associative_scan = builtin(tl_core.associative_scan)
assume = builtin(tl_core.assume)
atomic_add = builtin(tl_core.atomic_add)
atomic_and = builtin(tl_core.atomic_and)
atomic_cas = builtin(tl_core.atomic_cas)
atomic_max = builtin(tl_core.atomic_max)
atomic_min = builtin(tl_core.atomic_min)
atomic_or = builtin(tl_core.atomic_or)
atomic_xchg = builtin(tl_core.atomic_xchg)
atomic_xor = builtin(tl_core.atomic_xor)
broadcast = builtin(tl_core.broadcast)
cast = builtin(tl_core.cast)
clamp = builtin(tl_core.clamp)
device_assert = builtin(tl_core.device_assert)
device_print = builtin(tl_core.device_print)
expand_dims = builtin(tl_core.expand_dims)
gather = builtin(tl_core.gather)
inline_asm_elementwise = builtin(tl_core.inline_asm_elementwise)
join = builtin(tl_core.join)
load = builtin(tl_core.load)
map_elementwise = builtin(tl_core.map_elementwise)
max_constancy = builtin(tl_core.max_constancy)
max_contiguous = builtin(tl_core.max_contiguous)
maximum = builtin(tl_core.maximum)
minimum = builtin(tl_core.minimum)
mul = builtin(tl_core.mul)
multiple_of = builtin(tl_core.multiple_of)
num_programs = builtin(tl_core.num_programs)
permute = builtin(tl_core.permute)
program_id = builtin(tl_core.program_id)
reduce = builtin(tl_core.reduce)
reshape = builtin(tl_core.reshape)
split = builtin(tl_core.split)
static_assert = builtin(tl_core.static_assert)
static_print = builtin(tl_core.static_print)
store = builtin(tl_core.store)
sub = builtin(tl_core.sub)
to_tensor = builtin(tl_core.to_tensor)
expect_zero = builtin(tl_core.expect_zero)
where = builtin(tl_core.where)


class distributed_type(block_type):

    def __init__(self, element_ty: dtype, shape: List[int], layout):
        layout = _unwrap_if_constexpr(layout)
        shape = _unwrap_if_constexpr(shape)
        super().__init__(element_ty, shape)
        self.layout = layout
        self.name = f"<{self.shape}, {self.element_ty}, {self.layout}>"
        assert isinstance(layout, DistributedLayout), "tensor layout must be a DistributedLayout"
        if not isinstance(layout, (AutoLayout, CoalescedLayout)):
            assert len(
                shape
            ) == layout.rank, f"tensor shape and layout rank mismatch: shape={shape}, layout={layout}, shape rank={len(shape)}, layout rank={layout.rank}"

    def to_ir(self, builder: ir.builder) -> ir.type:
        elem_ty = self.element_ty.to_ir(builder)
        layout = self.layout._to_ir(builder)
        return builder.get_distributed_ty(elem_ty, self.shape, layout)

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = "_".join(map(str, self.shape))
        layout = self.layout.mangle()
        return f"{elt}S{shape}SL{layout}L"

    def with_element_ty(self, scalar_ty: dtype) -> block_type:
        return distributed_type(scalar_ty, self.shape, self.layout)

    def __eq__(self, other) -> bool:
        if not isinstance(other, distributed_type):
            return False
        return super().__eq__(other) and self.layout == other.layout


class shared_memory_descriptor_type(base_type):

    def __init__(self, element_ty, shape, layout, alloc_shape):
        shape = _unwrap_if_constexpr(shape)
        alloc_shape = _unwrap_if_constexpr(alloc_shape)
        layout = _unwrap_if_constexpr(layout)
        self.element_ty = element_ty
        self.shape = shape
        self.layout = layout
        self.alloc_shape = alloc_shape
        assert isinstance(layout, SharedLayout)

    def to_ir(self, builder: GluonOpBuilder) -> None:
        return builder.get_shared_mem_desc_ty(
            self.element_ty.to_ir(builder),
            self.shape,
            self.layout._to_ir(builder),
            self.alloc_shape,
        )

    def _unflatten_ir(self, handles: List[ir.Value], cursor: int) -> Tuple[shared_memory_descriptor, int]:
        value = shared_memory_descriptor(handles[cursor], self.element_ty, self.shape, self.layout, self.alloc_shape)
        return value, cursor + 1

    def _flatten_ir_types(self, builder: GluonOpBuilder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def __str__(self) -> str:
        return f"shared_memory_descriptor<{self.element_ty}, {self.shape}, {self.layout}, {self.alloc_shape}>"

    @property
    def nbytes_per_cta(self) -> int:
        if isinstance(self.layout, SharedLinearLayout):
            cga_layout = []
            dim_bases = [0] * len(self.shape)
            for basis in self.layout.block_bases:
                cga_basis = [0] * len(self.shape)
                for dim, value in enumerate(basis):
                    if value != 0:
                        cga_basis[dim] = 1 << dim_bases[dim]
                        dim_bases[dim] += 1
                cga_layout.append(cga_basis)
        else:
            cga_layout = self.layout.cga_layout
        shape_per_cta = _get_shape_per_cta(self.shape, cga_layout)
        return math.prod(shape_per_cta) * self.element_ty.primitive_bitwidth // 8

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.layout == other.layout
                and self.alloc_shape == other.alloc_shape)

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        shape_str = "_".join([str(s) for s in self.shape])
        alloc_shape_str = "_".join([str(s) for s in self.alloc_shape])
        return f"MD{self.element_ty.mangle()}S{shape_str}SL{self.layout.mangle()}LAS{alloc_shape_str}ASMD"


def _add_atomic_scatter_docstring(kind: str) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        integer_only = kind in ("max", "min", "logical and", "logical or", "logical xor")
        value_kind = "integer values" if integer_only else "values"
        value_type = ("Integer tensor broadcast-compatible with :code:`indices`"
                      if integer_only else "Tensor broadcast-compatible with :code:`indices`")
        docstr = f"""
    Performs an atomic scatter {kind} on this shared-memory descriptor.

    For each input position :code:`I`, reads from and writes to the element whose
    coordinate at :code:`axis` is replaced by :code:`indices[I]`:
      :code:`old = dst[I[0], ..., indices[I], ..., I[n]]`
      :code:`dst[I[0], ..., indices[I], ..., I[n]] = op(old, values[I])`
    where :code:`op` is {kind}.

    :code:`values`, :code:`indices`, and optional :code:`mask` are broadcast to a
    common tensor shape before the atomic operation. The returned tensor has that
    broadcasted shape. For example, with :code:`axis=1`, :code:`values` of shape
    :code:`[N, 1]`, and :code:`indices` of shape :code:`[1, M]`, the operation
    behaves as if both operands had shape :code:`[N, M]`:
      :code:`old[i, j] = dst[i, indices[0, j]]`
      :code:`dst[i, indices[0, j]] = op(old[i, j], values[i, 0])`
    A :code:`mask` of shape :code:`[N, 1]` would also broadcast over :code:`M`.

    Return the data stored at the scattered location before the atomic operation.

    :param values: The {value_kind} with which to perform the atomic operation
    :type values: {value_type}
    :param indices: The indices to update along :code:`axis`
    :type indices: Integer tensor
    :param axis: The axis along which to update values
    :type axis: int
    :param mask: Boolean tensor broadcast-compatible with :code:`values` and :code:`indices`,
        selecting which elements to update
    :type mask: Tensor, optional

    :note: This operation currently uses relaxed memory semantics. Users are responsible
        for inserting mbarrier synchronization themselves.
    """
        func.__doc__ = docstr
        return func

    return _decorator


class shared_memory_descriptor(base_value):
    """
    Represents a handle to a shared memory allocation in Gluon IR.
    """

    def __init__(self, handle, element_ty, shape, layout, alloc_shape):
        self.handle = handle
        self.type = shared_memory_descriptor_type(element_ty, shape, layout, alloc_shape)

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def dtype(self):
        return self.type.element_ty

    @property
    def shape(self):
        return self.type.shape

    @property
    def rank(self):
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes_per_cta(self) -> int:
        return self.type.nbytes_per_cta

    @property
    def layout(self):
        return self.type.layout

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def load(self, layout, _semantic: GluonSemantic = None) -> tensor:
        """
        Load a tensor from shared memory.

        Args:
            layout (DistributedLayout): The destination layout of the tensor.

        Returns:
            tensor: A Gluon tensor containing the loaded data.
        """
        layout = _unwrap_if_constexpr(layout)
        return _semantic.shared_load(self, layout)

    @builtin
    def store(self, value, _semantic: GluonSemantic = None) -> None:
        """
        Store a tensor into shared memory.

        Args:
            value (tensor): The tensor whose contents to store.
        """
        return _semantic.shared_store(self, value)

    @builtin
    def gather(self, indices, axis, _semantic: GluonSemantic = None) -> tensor:
        """
        Gather elements from shared memory along a specified axis using an indices tensor.

        For each output position I, the operation reads from src where the coordinate at
        the gather axis is replaced by indices[I]:
          result[I] = src[I[0], ..., indices[I], ..., I[n]]

        Args:
            indices (tensor): Tensor specifying which indices to gather along the axis.
            axis (int): The axis along which to gather values.

        Returns:
            tensor: Gluon tensor with the gathered elements (same shape as indices).
        """
        indices = _unwrap_if_constexpr(indices)
        axis = _unwrap_if_constexpr(axis)
        return _semantic.shared_gather(self, indices, axis)

    @builtin
    def scatter(self, values, indices, axis, _semantic: GluonSemantic = None):
        """
        Scatter elements to shared memory along a specified axis using an indices tensor.

        For each input position I, the operation writes to dst where the coordinate at
        the scatter axis is replaced by indices[I]:
          dst[I[0], ..., indices[I], ..., I[n]] = values[I]

        Broadcasting:
            values and indices are broadcast to a common tensor shape before the scatter.
            For example, with axis=1, values of shape [N, 1] and indices of shape
            [1, M] behave as if both operands had shape [N, M]:
              dst[i, indices[0, j]] = values[i, 0]

        Args:
            values (tensor): Tensor with values to scatter (broadcast-compatible with indices).
            indices (tensor): Tensor specifying which indices to scatter to along the axis.
            axis (int): The axis along which to scatter values.
        """
        values = _unwrap_if_constexpr(values)
        indices = _unwrap_if_constexpr(indices)
        axis = _unwrap_if_constexpr(axis)
        return _semantic.shared_scatter(self, values, indices, axis)

    def _atomic_scatter_rmw(self, op, values, indices, axis, mask, _semantic: GluonSemantic = None) -> tensor:
        values = _unwrap_if_constexpr(values)
        indices = _unwrap_if_constexpr(indices)
        axis = _unwrap_if_constexpr(axis)
        mask = _unwrap_if_constexpr(mask)
        if mask is not None:
            mask = _semantic.to_tensor(mask)
        return _semantic.shared_atomic_scatter_rmw(self, op, values, indices, axis, mask)

    @builtin
    @_add_atomic_scatter_docstring("add")
    def atomic_scatter_add(self, values, indices, axis, mask=None, _semantic: GluonSemantic = None) -> tensor:
        return self._atomic_scatter_rmw("add", values, indices, axis, mask, _semantic)

    @builtin
    @_add_atomic_scatter_docstring("max")
    def atomic_scatter_max(self, values, indices, axis, mask=None, _semantic: GluonSemantic = None) -> tensor:
        return self._atomic_scatter_rmw("max", values, indices, axis, mask, _semantic)

    @builtin
    @_add_atomic_scatter_docstring("min")
    def atomic_scatter_min(self, values, indices, axis, mask=None, _semantic: GluonSemantic = None) -> tensor:
        return self._atomic_scatter_rmw("min", values, indices, axis, mask, _semantic)

    @builtin
    @_add_atomic_scatter_docstring("logical and")
    def atomic_scatter_and(self, values, indices, axis, mask=None, _semantic: GluonSemantic = None) -> tensor:
        return self._atomic_scatter_rmw("and", values, indices, axis, mask, _semantic)

    @builtin
    @_add_atomic_scatter_docstring("logical or")
    def atomic_scatter_or(self, values, indices, axis, mask=None, _semantic: GluonSemantic = None) -> tensor:
        return self._atomic_scatter_rmw("or", values, indices, axis, mask, _semantic)

    @builtin
    @_add_atomic_scatter_docstring("logical xor")
    def atomic_scatter_xor(self, values, indices, axis, mask=None, _semantic: GluonSemantic = None) -> tensor:
        return self._atomic_scatter_rmw("xor", values, indices, axis, mask, _semantic)

    @builtin
    @_add_atomic_scatter_docstring("exchange")
    def atomic_scatter_xchg(self, values, indices, axis, mask=None, _semantic: GluonSemantic = None) -> tensor:
        return self._atomic_scatter_rmw("xchg", values, indices, axis, mask, _semantic)

    def slice(self, start, length, dim=0, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        """
        Create a subview of shared memory by slicing along a given dimension.

        Args:
            start (int): The starting index of the slice.
            length (int): The length of the slice.
            dim (int): The dimension to slice (default: 0).

        Returns:
            shared_memory_descriptor: Descriptor for the sliced subview.
        """
        start = _unwrap_if_constexpr(start)
        length = _unwrap_if_constexpr(length)
        dim = _unwrap_if_constexpr(dim)
        return _semantic.memdesc_slice(self, start, length, dim)

    @builtin
    def index(self, index, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        """
        Create a subview of shared memory by indexing along the first dimension.

        Args:
            index (int): The index at which to take the subview.

        Returns:
            shared_memory_descriptor: Descriptor for the indexed subview.
        """
        index = _unwrap_if_constexpr(index)
        return _semantic.memdesc_index(self, index)

    @builtin
    def permute(self, order, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        """
        Permute the dimensions of the shared memory descriptor.

        Args:
            order (List[int]): The new ordering of dimensions.

        Returns:
            shared_memory_descriptor: Descriptor with permuted dimensions.
        """
        order = [_unwrap_if_constexpr(o) for o in order]
        return _semantic.memdesc_trans(self, order)

    @builtin
    def reshape(self, shape, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        """
        Reshape the shared memory descriptor to a new shape and layout.

        Args:
            shape (List[int]): The target shape.

        Returns:
            shared_memory_descriptor: Descriptor with the new shape and layout.
        """
        shape = [_unwrap_if_constexpr(s) for s in shape]

        return _semantic.memdesc_reshape(self, shape)

    @builtin
    def _reinterpret(self, dtype=None, shape=None, layout=None,
                     _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        """
        Reinterpret the shared memory descriptor as a different dtype, shape, or layout.

        Args:
            dtype (dtype): The new data type. Defaults to the descriptor dtype.
            shape (List[int]): The new shape. Defaults to the descriptor shape.
            layout (SharedLayout): The new layout. Defaults to the descriptor layout.

        Returns:
            shared_memory_descriptor: Descriptor with updated type and layout.
        """
        dtype = self.dtype if dtype is None else _unwrap_if_constexpr(dtype)
        shape = self.shape if shape is None else [_unwrap_if_constexpr(s) for s in shape]
        layout = self.layout if layout is None else _unwrap_if_constexpr(layout)

        return _semantic.memdesc_reinterpret(self, dtype, shape, layout)

    @builtin
    def _keep_alive(self, _semantic: GluonSemantic = None) -> None:
        """
        Dummy use to keep the shared memory descriptor alive.
        """
        return _semantic.shared_dealloc(self)


@builtin
def arange(start, end, layout=None, _semantic=None):
    """
    Generate a sequence tensor with values in [start, end) using a specified layout.

    Args:
        start (int): Inclusive start of the sequence.
        end (int): Exclusive end of the sequence.
        layout (DistributedLayout): The layout of the output tensor. Defaults to AutoLayout.

    Returns:
        tensor: A 1D tensor containing sequential values.
    """
    start = _unwrap_if_constexpr(start)
    end = _unwrap_if_constexpr(end)
    layout = _unwrap_if_constexpr(layout)
    return _semantic.arange(start, end, layout)


@builtin
def convert_layout(value, layout, assert_trivial=False, _semantic=None):
    """
    Convert a tensor to a different distributed layout.

    Args:
        value (tensor): The input tensor.
        layout (DistributedLayout): The target layout.
        assert_trivial (bool): If True, asserts that the conversion is trivial (no data movement).

    Returns:
        tensor: The tensor with the new layout.
    """
    layout = _unwrap_if_constexpr(layout)
    return _semantic.convert_layout(value, layout, assert_trivial)


@builtin
def full(shape, value, dtype, layout=None, _semantic=None):
    """
    Create a tensor filled with a scalar value, with specified shape, dtype, and layout.

    Args:
        shape (Sequence[int]): The shape of the tensor.
        value (int or float): The fill value.
        dtype (dtype): The data type for the tensor.
        layout (Optional[DistributedLayout]): The layout of the output tensor, defaults to AutoLayout().

    Returns:
        tensor: A tensor where every element equals value.
    """
    shape = _unwrap_shape(shape)
    value = _unwrap_if_constexpr(value)
    dtype = _unwrap_if_constexpr(dtype)
    layout = _unwrap_if_constexpr(layout)
    return _semantic.full(shape, value, dtype, layout)


@builtin
def histogram(input, num_bins, mask=None, layout=None, _semantic=None, _generator=None):
    """
    Compute a histogram of a 1D integer tensor.

    Args:
        input (tensor): 1D tensor of integer values.
        num_bins (int): Number of bins. Bins have width 1 and start at 0.
        mask (Optional[tensor]): Boolean mask to exclude elements when False.
        layout (DistributedLayout): Destination layout of the output histogram.

    Returns:
        tensor: 1D int32 tensor of length `num_bins` with the requested layout.
    """
    num_bins = _unwrap_if_constexpr(num_bins)
    layout = _unwrap_if_constexpr(layout)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    return _semantic.histogram(input, num_bins, mask, layout)


@builtin
def allocate_shared_memory(element_ty, shape, layout, value=None, _semantic=None) -> shared_memory_descriptor:
    """
    Allocate shared memory for a tensor with the given element type, shape, and layout.

    Args:
        element_ty (dtype): The element data type.
        shape (Sequence[int]): The dimensions of the shared memory.
        layout (SharedLayout): The shared memory layout.
        value (tensor, optional): Initial value to copy into shared memory.

    Returns:
        shared_memory_descriptor: Descriptor for the allocated memory.
    """
    element_ty = _unwrap_if_constexpr(element_ty)
    shape = _unwrap_if_constexpr(shape)
    shape = [_unwrap_if_constexpr(s) for s in shape]
    layout = _unwrap_if_constexpr(layout)
    return _semantic.allocate_shared(element_ty, shape, layout, value)


@builtin
def set_auto_layout(value, layout, _semantic=None):
    """
    Set a tensor with AutoLayout to a concrete layout

    Args:
        value (tensor): The input tensor.
        layout (DistribtedLayout): The target layout.

    Returns:
        tensor: The tensor with the new layout.
    """
    layout = _unwrap_if_constexpr(layout)
    return _semantic.set_auto_layout(value, layout)


@builtin
def fp4_to_fp(src, elem_type, axis, _semantic=None):
    """
    Upcast a tensor from fp4 (e2m1) to another floating point type.
    """
    axis = _unwrap_if_constexpr(axis)
    elem_type = _unwrap_if_constexpr(elem_type)
    return _semantic.fp4_to_fp(src, elem_type, axis)


@builtin
def warp_specialize(functions_and_args, worker_num_warps, worker_num_regs=None, _semantic=None, _generator=None):
    """
    Create a warp-specialized execution region, partitioning work across warps.

    This forks the current execution into a "default partition" and an arbitrary number of
    "worker partitons". The default partition is executed in the same :code:`num_warps` warps as
    the parent region, and may accept tensor arguments and return tensors. Worker partitions are
    executed in additional warps, which sit idle while executing the parent region.

    Note that calling warp_specialize recursively is not supported.

    Args:
        functions_and_args (List[Tuple[Callable, Any]]): List of functions and arguments for each partition. The first of which is the default partition.
        worker_num_warps (List[int]): Number of warps used for each worker partition.
        worker_num_regs (List[int], optional): Number of registers for each worker partition.
            If not None, will be used by backend for dynamic register reallocation.

    Returns:
        Tuple[Any, ...]: Results from the default partition.
    """
    worker_num_warps = [_unwrap_if_constexpr(w) for w in worker_num_warps]
    if worker_num_regs is not None:
        worker_num_regs = [_unwrap_if_constexpr(r) for r in worker_num_regs]
    return _semantic.warp_specialize(functions_and_args, worker_num_warps, worker_num_regs, _generator)


@builtin
def num_warps(_semantic=None, _generator=None):
    """
    Returns the number of warps that execute the current context, including in warp-specialized regions.
    """
    return _semantic.num_warps(_generator)


@builtin
def num_ctas(_semantic=None):
    """
    Returns the number of CTAs in the current kernel
    """
    return _semantic.num_ctas()


@builtin
def barrier(*, cluster: bool = False, _semantic=None):
    """
    Insert a barrier to synchronize threads within a CTA, or across a cluster.

    Args:
        cluster (bool): Whether to synchronize across the CTA cluster.
    """
    cluster = _unwrap_if_constexpr(cluster)
    num_ctas = _unwrap_if_constexpr(_semantic.num_ctas())
    if num_ctas == 1 or not cluster:
        return _semantic.debug_barrier()
    _semantic.builder.create_cluster_barrier()


@builtin
def bank_conflicts(distr_ty, shared_ty, _semantic=None) -> int:
    """
    Count the bank conflicts per wavefront of each instruction generated when
    reading/writing the distributed tensor from/to the shared memory descriptor
    using ld.shared/st.shared instructions.

    We define a bank conflict of N to be the excess number of memory accesses that each
    wavefront needs to access the shared memory descriptor. When one uses no ld/st
    vectorization, this is equal to t he number of excess memory accesses per instruction.

    Args:
        distr_ty (distributed_type): The distributed tensor.
        shared_ty (shared_memory_descriptor_type): The shared memory descriptor.

    Returns:
        int: The number of bank conflicts.
    """
    distr_ty = _unwrap_if_constexpr(distr_ty)
    shared_ty = _unwrap_if_constexpr(shared_ty)
    return _semantic.bank_conflicts(distr_ty, shared_ty)


@builtin
def to_linear_layout(layout, shape, _semantic=None):
    layout = _unwrap_if_constexpr(layout)
    shape = _unwrap_shape(shape)
    return _semantic.to_linear_layout(layout, shape)


@builtin
def dot_fma(a, b, acc, _semantic=None):
    assert isinstance(a, tensor), "a must be a tensor"
    assert isinstance(b, tensor), "b must be a tensor"
    assert isinstance(acc, tensor), "acc must be a tensor"

    mma_layout = acc.type.layout
    assert isinstance(mma_layout, BlockedLayout), "acc must have a BlockedLayout"
    assert isinstance(a.type.layout, DotOperandLayout), "a must have a DotOperandLayout"
    assert isinstance(b.type.layout, DotOperandLayout), "b must have a DotOperandLayout"
    assert a.type.layout.parent == mma_layout, "a's parent layout must be the same as acc's layout"
    assert b.type.layout.parent == mma_layout, "b's parent layout must be the same as acc's layout"
    assert a.type.layout.operand_index == 0, "a's operand index must be 0"
    assert b.type.layout.operand_index == 1, "b's operand index must be 1"
    assert len(acc.shape) == 2 or len(acc.shape) == 3
    assert len(acc.shape) == len(a.shape) == len(b.shape)

    unified_dot_shape = acc.shape + a.shape[-1:]  # join batch/M/N and K in one list
    if math.prod(unified_dot_shape) > 2**19:
        dot_name = "batched dot" if len(acc.shape) == 3 else "dot"
        shape_str = "x".join([str(x) for x in unified_dot_shape])
        warnings.warn(f"Large {dot_name} FMA instruction size {shape_str} may have slow compile times")

    handle = _semantic.dot(a, b, acc, input_precision=None, max_num_imprecise_acc=None, out_dtype=acc.dtype).handle
    return tensor(handle, acc.type)
<|endoftext|>from ._core import (
    aggregate_replace,
    base_value,
    base_type,
    block_type,
    broadcast,
    cast,
    clamp,
    constexpr,
    dtype,
    void,
    int1,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float8e5,
    float8e5b16,
    float8e4nv,
    float8e4b8,
    float8e4b15,
    float16,
    bfloat16,
    float32,
    float64,
    pointer_type,
    shared_memory_descriptor,
    tensor,
    tuple,
    tuple_type,
    _unwrap_if_constexpr,
    # API Functions
    add,
    allocate_shared_memory,
    arange,
    associative_scan,
    assume,
    atomic_add,
    atomic_and,
    atomic_cas,
    atomic_max,
    atomic_min,
    atomic_or,
    atomic_xchg,
    atomic_xor,
    bank_conflicts,
    convert_layout,
    device_assert,
    device_print,
    dot_fma,
    expand_dims,
    full,
    fp4_to_fp,
    gather,
    num_warps,
    num_ctas,
    histogram,
    inline_asm_elementwise,
    join,
    load,
    map_elementwise,
    max_constancy,
    max_contiguous,
    maximum,
    minimum,
    mul,
    multiple_of,
    num_programs,
    permute,
    program_id,
    reduce,
    reshape,
    distributed_type,
    shared_memory_descriptor_type,
    set_auto_layout,
    split,
    static_assert,
    static_print,
    static_range,
    store,
    sub,
    barrier,
    to_linear_layout,
    to_tensor,
    expect_zero,
    warp_specialize,
    where,
)
from ._layouts import (
    AutoLayout,
    BlockedLayout,
    SliceLayout,
    DistributedLinearLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    NVMMASharedLayout,
    SwizzledSharedLayout,
    PaddedSharedLayout,
    SharedLinearLayout,
    CoalescedLayout,
)
from ._math import (
    umulhi,
    exp,
    exp2,
    fma,
    log,
    log2,
    cos,
    rsqrt,
    sin,
    sqrt,
    sqrt_rn,
    abs,
    fdiv,
    div_rn,
    erf,
    floor,
    ceil,
)
from ._standard import (
    cdiv,
    full_like,
    max,
    min,
    ravel,
    reduce_or,
    sum,
    xor_sum,
    zeros,
    zeros_like,
)

from . import nvidia
from . import amd
from . import extra
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass

from triton.experimental.gluon.language._core import _unwrap_if_constexpr, builtin

from ..blackwell import (
    TensorMemoryLayout,
    _TensorMemoryLinearLayout,
    _packed_arith,
    add2,
    allocate_tensor_memory,
    async_copy,
    async_store,
    clc,
    fence_async_shared,
    fma2,
    max2,
    min2,
    mma_v2,
    mul2,
    sub2,
    tensor_memory_descriptor,
    tensor_memory_descriptor_type,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
    tcgen05_mma_scaled,
    tma,
)
from ..blackwell import TensorMemoryScalesLayout as _BlackwellTensorMemoryScalesLayout
from . import mbarrier

__all__ = [
    "add2",
    "add4",
    "allocate_tensor_memory",
    "async_copy",
    "async_store",
    "clc",
    "fence_async_shared",
    "fma2",
    "fma4",
    "mbarrier",
    "max2",
    "min2",
    "mma_v2",
    "mul2",
    "mul4",
    "sub2",
    "sub4",
    "tensor_memory_descriptor",
    "tensor_memory_descriptor_type",
    "tcgen05_commit",
    "tcgen05_copy",
    "tcgen05_mma",
    "tcgen05_mma_barrier_count",
    "tcgen05_mma_scaled",
    "TensorMemoryLayout",
    "TensorMemoryScalesLayout",
    "tma",
    "_TensorMemoryLinearLayout",
]


@builtin
def add4(lhs, rhs, dtype=None, _semantic=None):
    """Add four-lane FP8 or FP4 operands with a Rubin packed instruction."""
    return _packed_arith("add", (lhs, rhs), dtype, _semantic)


@builtin
def sub4(lhs, rhs, dtype=None, _semantic=None):
    """Subtract four-lane FP8 or FP4 operands with a Rubin packed instruction."""
    return _packed_arith("sub", (lhs, rhs), dtype, _semantic)


@builtin
def mul4(lhs, rhs, dtype=None, _semantic=None):
    """Multiply four-lane FP8 or FP4 operands with a Rubin packed instruction."""
    return _packed_arith("mul", (lhs, rhs), dtype, _semantic)


@builtin
def fma4(lhs, rhs, acc, dtype=None, _semantic=None):
    """Perform a Rubin four-lane packed FP8 or FP4 fused multiply-add."""
    return _packed_arith("fma", (lhs, rhs, acc), dtype, _semantic)


@dataclass(frozen=True, eq=True)
class TensorMemoryScalesLayout(_BlackwellTensorMemoryScalesLayout):
    """
    Describes the layout for tensor memory scales in Rubin architecture.

    Args:
        cga_layout (Optional[List[List[int]]]): CGA layout bases. Defaults to [].
        block_rep_order (str): Order of repeated scale blocks. Must be either
            ``"mnThenK"`` or ``"kThenMn"``. Defaults to ``"mnThenK"``.
    """
    block_rep_order: str = "mnThenK"

    def __post_init__(self):
        super().__post_init__()
        super().__setattr__("block_rep_order", _unwrap_if_constexpr(self.block_rep_order))
        assert self.block_rep_order in ("mnThenK", "kThenMn")

    def _to_ir(self, builder):
        return builder.get_tensor_memory_scales_layout([list(basis) for basis in self.cga_layout], self.block_rep_order)

    def mangle(self) -> str:
        cga_layout_str = "_".join("~".join(map(str, basis)) for basis in self.cga_layout)
        return f"TLS{self.block_rep_order}_{cga_layout_str}TLS"

    def __hash__(self):
        return hash((tuple(tuple(b) for b in self.cga_layout), self.block_rep_order))
<|endoftext|>from ..._core import _unwrap_if_constexpr, builtin
from ..hopper.mbarrier import (
    MBarrierLayout,
    allocate_mbarrier,
    expect,
    fence_init_release_cluster,
    init,
    invalidate,
    wait,
)

__all__ = [
    "allocate_mbarrier",
    "arrive",
    "expect",
    "fence_init_release_cluster",
    "init",
    "invalidate",
    "MBarrierLayout",
    "wait",
]


@builtin
def arrive(mbarrier, *, count=1, cta_mask=0, pred=True, _semantic=None):
    """
    Arrive at an mbarrier with a specified count.

    When ``cta_mask`` is non-zero, the arrive is multicast across the cluster.
    Each bit set in the mask identifies a CTA ID dimension to multicast
    along. CTA IDs ``a`` and ``b`` belong to the same equivalence class iff
    ``a & ~cta_mask == b & ~cta_mask``; all CTAs in a class multicast to each
    other. Multicast requires ``num_ctas > 1``, ``0 < cta_mask <= num_ctas - 1``,
    and the barrier must have the identity CGA layout ``[[1], [2], ...]``. The
    default value of ``cta_mask`` is 0 (no multicast).

    Args:
        mbarrier (shared_memory_descriptor): Barrier to be signalled.
        count (int): Count to arrive with. Defaults to 1.
        cta_mask (int): CTA broadcast dimension bits (see above). Defaults
            to 0 (no multicast). Must satisfy ``0 < cta_mask <= num_ctas - 1``
            when non-zero.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    count = _unwrap_if_constexpr(count)
    cta_mask = _unwrap_if_constexpr(cta_mask)
    if not isinstance(cta_mask, int) or isinstance(cta_mask, bool):
        raise TypeError(f"cta_mask must be an int, got {type(cta_mask).__name__}")
    if cta_mask:
        num_ctas = _semantic.builder.options.num_ctas
        if cta_mask < 0:
            raise ValueError(f"cta_mask must be positive, got {cta_mask}")
        if cta_mask > num_ctas - 1:
            raise ValueError(f"cta_mask must be <= num_ctas - 1 ({num_ctas - 1}), got {cta_mask}")
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, cta_mask, pred.handle)
<|endoftext|>from __future__ import annotations

from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["arrive", "wait", "barrier"]


@builtin
def arrive(relaxed: bool = False, _semantic=None):
    """
    Arrive at a barrier that synchronizes across the CTA cluster.

    Args:
        relaxed (bool): Whether to use relaxed semantics. Defaults to False.
    """
    relaxed = _unwrap_if_constexpr(relaxed)
    _semantic.builder.create_cluster_arrive(relaxed)


@builtin
def wait(_semantic=None):
    """
    Wait for all CTAs in the cluster to arrive at the cluster barrier.
    """
    _semantic.builder.create_cluster_wait()


@builtin
def barrier(relaxed: bool = False, _semantic=None):
    """
    Barrier that synchronizes across the CTA cluster.

    Args:
        relaxed (bool): Whether to use relaxed arrival semantics. Defaults to
            False.
    """
    relaxed = _unwrap_if_constexpr(relaxed)
    _semantic.builder.create_cluster_barrier(relaxed)
<|endoftext|>from __future__ import annotations
from triton.compiler.code_generator import unflatten_ir_values
from ..ampere import async_copy, mma_v2
from . import cluster, mbarrier, tma
from ... import _core

from typing import List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from triton._C.libtriton import ir

__all__ = [
    "async_copy",
    "async_store",
    "cluster",
    "fence_async_shared",
    "mbarrier",
    "mma_v2",
    "tma",
    "warpgroup_mma",
    "warpgroup_mma_wait",
]


def _check(cond, msg_fn, category=ValueError):
    if not cond:
        raise category(msg_fn())


@_core.builtin
def fence_async_shared(cluster=False, _semantic=None):
    """
    Issue a fence to complete asynchronous shared memory operations.

    Args:
        cluster (bool): Whether to fence across cluster. Defaults to False.
    """
    cluster = _core._unwrap_if_constexpr(cluster)
    _semantic.builder.create_fence_async_shared(cluster)


@_core.builtin
def async_store(dst, value, mbarrier, _semantic=None):
    """
    Store a tensor to shared memory asynchronously and signal an mbarrier on completion.
    Requires a CTA cluster with at least two CTAs.

    Args:
        dst (shared_memory_descriptor): Destination shared memory descriptor.
        value (tensor): Tensor whose contents to store.
        mbarrier (shared_memory_descriptor): Barrier signaled when the store completes.
    """
    _check(isinstance(value, _core.tensor), lambda: f"expected 'value' to be a tensor, but got a {type(value)}")
    _check(isinstance(mbarrier, _core.shared_memory_descriptor),
           lambda: f"expected 'mbarrier' to be a shared_memory_descriptor, but got a {type(mbarrier)}")
    _check(value.shape == dst.shape, lambda: f"source shape {value.shape} and destination shape {dst.shape} must match")
    _check(value.dtype == dst.dtype, lambda: f"source dtype {value.dtype} and destination dtype {dst.dtype} must match")
    _semantic.builder.create_async_shared_store(dst.handle, value.handle, mbarrier.handle)


class warpgroup_mma_accumulator_type(_core.base_type):
    tensor_type: _core.dtype

    def __init__(self, tensor_type: _core.dtype):
        self.tensor_type = tensor_type

    def __str__(self) -> str:
        return f"warpgroup_mma_accumulator<{self.tensor_type}>"

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[warpgroup_mma_accumulator, int]:
        return warpgroup_mma_accumulator(handles[cursor], self.tensor_type), cursor + 1

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        self.tensor_type._flatten_ir_types(builder, out)

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.tensor_type == other.tensor_type

    def mangle(self) -> str:
        return f"FT{self.tensor_type.mangle()}FT"


class warpgroup_mma_accumulator(_core.base_value):
    handle: ir.value
    type: warpgroup_mma_accumulator_type

    def __init__(self, handle, tensor_type: _core.dtype):
        self.handle = handle
        self.type = warpgroup_mma_accumulator_type(tensor_type)

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)


@_core.builtin
def warpgroup_mma_init(value, _semantic=None):
    assert isinstance(value, _core.tensor)
    return warpgroup_mma_accumulator(value.handle, value.type)


@_core.builtin
def warpgroup_mma(a, b, acc, *, use_acc=True, precision=None, max_num_imprecise_acc=None, is_async=False,
                  _semantic=None):
    """
    Perform warpgroup MMA (Tensor Core) operations.
    acc = a * b + (acc if use_acc else 0)

    Args:
        a (tensor or shared_memory_descriptor): Left hand side operand.
        b (shared_memory_descriptor): Right hand side operand.
        acc (tensor): Accumulator tensor.
        use_acc (bool): Whether to use the initial value of the accumulator. Defaults to True.
        precision (str, optional): Dot input precision. Defaults to builder default.
        max_num_imprecise_acc (int): Max imprecise accumulations. Used for fp8 -> fp32 dot. Determines how many accumulation are done in limited precision. Defaults to None, which means no upcasting is done.
        is_async (bool): Whether operation is asynchronous. Defaults to False.

    Returns:
        tensor or warpgroup_mma_accumulator: Returns the result if synchronous, or a token to load the value once computed if asynchronous.
    """
    use_acc = _semantic.to_tensor(use_acc)

    if precision is None:
        precision = _semantic.builder.options.default_dot_input_precision
    precision = _core._unwrap_if_constexpr(precision)

    precision = _semantic._str_to_dot_input_precision(precision)

    K = a.type.shape[-1]
    if max_num_imprecise_acc is None:
        if a.dtype.is_fp8() and b.dtype.is_fp8():
            max_num_imprecise_acc = _semantic.builder.options.max_num_imprecise_acc_default
        else:
            max_num_imprecise_acc = 0
    else:
        if a.dtype.is_fp8() and b.dtype.is_fp8() and max_num_imprecise_acc > K:
            raise ValueError(f"max_num_imprecise_acc ({max_num_imprecise_acc}) must be <= K ({K})")

    max_num_imprecise_acc = _core._unwrap_if_constexpr(max_num_imprecise_acc)
    is_async = _core._unwrap_if_constexpr(is_async)

    handle = _semantic.builder.create_warpgroup_mma(a.handle, b.handle, acc.handle, use_acc.handle, precision,
                                                    max_num_imprecise_acc, is_async)
    tensor_ty = acc.type.tensor_type if isinstance(acc, warpgroup_mma_accumulator) else acc.type
    if is_async:
        return warpgroup_mma_accumulator(handle, tensor_ty)
    else:
        return _core.tensor(handle, tensor_ty)


@_core.builtin
def warpgroup_mma_wait(num_outstanding=0, deps=None, _semantic=None):
    """
    Wait until `num_outstanding` or less warpgroup MMA operations are in-flight.

    Args:
        num_outstanding (int): Number of outstanding warpgroup MMA operations to wait for. Defaults to 0.
        deps (Sequence[tensor]): List of dependencies that need to be kept alive while the mma is unfinished.
    """
    if deps is None:
        raise ValueError("warpgroup_mma_wait deps must be given")
    deps_handles = [x.handle for x in deps] if deps is not None else []
    num_outstanding = _core._unwrap_if_constexpr(num_outstanding)
    results = _semantic.builder.create_warpgroup_mma_wait(deps_handles, num_outstanding)
    result_types = [dep.type.tensor_type if isinstance(dep, warpgroup_mma_accumulator) else dep.type for dep in deps]
    results = unflatten_ir_values(results, result_types)
    if len(deps) == 1:
        return next(results)
    return tuple(results)
<|endoftext|>from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from triton.language.core import base_type, base_value
import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._layouts import NVMMASharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

if TYPE_CHECKING:
    from triton._C import ir

__all__ = [
    "async_atomic_add",
    "async_atomic_and",
    "async_atomic_max",
    "async_atomic_min",
    "async_atomic_or",
    "async_atomic_xor",
    "async_copy_global_to_shared",
    "async_copy_global_to_shared_im2col",
    "async_copy_shared_to_global",
    "async_load",
    "async_load_im2col",
    "async_store",
    "store_wait",
    "tensor_descriptor",
    "tensor_descriptor_im2col",
    "tensor_descriptor_type",
    "tensor_descriptor_im2col_type",
    "make_tensor_descriptor",
]


@dataclass(eq=True)
class _tensor_descriptor_type_base(base_type):
    """Base class for tensor descriptor types (tiled and im2col)."""
    block_type: ttgl.block_type
    shape_type: ttgl.tuple_type
    strides_type: ttgl.tuple_type
    layout: NVMMASharedLayout

    # Subclasses must override these
    _type_name: str = ""
    _mangle_prefix: str = ""

    def __str__(self) -> str:
        return f"{self._type_name}<{self.block_type}, {self.layout}>"

    @property
    def nbytes_per_cta(self) -> int:
        cga_layout = self.layout.cga_layout
        if len(cga_layout) == 0:
            return self.block_type.nbytes
        num_cta_splits = 2**sum(any(x != 0 for x in basis) for basis in cga_layout)
        return self.block_type.nbytes // num_cta_splits

    def _to_ir(self, builder: ir.builder) -> ir.type:
        raise NotImplementedError

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        raise NotImplementedError

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self._to_ir(builder))
        self.shape_type._flatten_ir_types(builder, out)
        self.strides_type._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        return f"{self._mangle_prefix}{self.block_type.mangle()}_{self.layout.mangle()}{self._mangle_prefix}"


@dataclass(eq=True)
class tensor_descriptor_type(_tensor_descriptor_type_base):
    """Type for tiled tensor descriptors."""
    _type_name: str = "tensor_descriptor"
    _mangle_prefix: str = "TD"

    def _to_ir(self, builder: ir.builder) -> ir.type:
        is_signed = self.block_type.element_ty.is_int_signed()
        return builder.get_tensor_descriptor_layout_type(self.block_type.to_ir(builder), is_signed,
                                                         self.layout._to_ir(builder))

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        value = tensor_descriptor(handle, shape, strides, self.block_type, layout=self.layout)
        return value, cursor


@dataclass(eq=True)
class tensor_descriptor_im2col_type(_tensor_descriptor_type_base):
    """Type for im2col tensor descriptors (convolution-friendly access patterns)."""
    _type_name: str = "tensor_descriptor_im2col"
    _mangle_prefix: str = "TDI"

    def _to_ir(self, builder: ir.builder) -> ir.type:
        is_signed = self.block_type.element_ty.is_int_signed()
        return builder.get_tensor_descriptor_im2col_layout_type(self.block_type.to_ir(builder), is_signed,
                                                                self.layout._to_ir(builder))

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        value = tensor_descriptor_im2col(handle, shape, strides, self.block_type, layout=self.layout)
        return value, cursor


class _tensor_descriptor_value_base(base_value):

    def __init__(self, handle, shape: List[ttgl.tensor], strides: List[ttgl.tensor], block_type: ttgl.block_type,
                 layout: NVMMASharedLayout, type_cls):
        self.handle = handle
        self.shape = ttgl.tuple(shape)
        self.strides = ttgl.tuple(strides)
        self.type = type_cls(block_type, shape_type=self.shape.type, strides_type=self.strides.type, layout=layout)

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))
        self.shape._set_name(builder, name + ".shape")
        self.strides._set_name(builder, name + ".stride")

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)
        self.shape._flatten_ir(handles)
        self.strides._flatten_ir(handles)

    @property
    def nbytes_per_cta(self):
        return self.type.nbytes_per_cta

    @property
    def block_type(self):
        return self.type.block_type

    @property
    def block_shape(self):
        return self.type.block_type.shape

    @property
    def dtype(self):
        return self.type.block_type.element_ty

    @property
    def layout(self):
        return self.type.layout


class tensor_descriptor(_tensor_descriptor_value_base):

    def __init__(self, handle, shape: List[ttgl.tensor], strides: List[ttgl.tensor], block_type: ttgl.block_type,
                 layout: NVMMASharedLayout):
        super().__init__(handle, shape, strides, block_type, layout, tensor_descriptor_type)


class tensor_descriptor_im2col(_tensor_descriptor_value_base):

    def __init__(self, handle, shape: List[ttgl.tensor], strides: List[ttgl.tensor], block_type: ttgl.block_type,
                 layout: NVMMASharedLayout):
        super().__init__(handle, shape, strides, block_type, layout, tensor_descriptor_im2col_type)


def _emit_alignment_check(desc, coord, fn_name: str, arg_name: str, _semantic=None):
    coord = list(coord)[-1]
    align_bytes = 16
    if desc.layout.fp4_padded:
        align_bytes = 64
    dtype = desc.dtype
    assert dtype.primitive_bitwidth % 8 == 0, f"unexpected sub-byte dtype {dtype}"
    elem_bytes = dtype.primitive_bitwidth // 8
    align = align_bytes // elem_bytes

    align_val = ttgl.to_tensor(align, _semantic=_semantic)
    zero = ttgl.to_tensor(0, _semantic=_semantic)

    coord = ttgl.to_tensor(coord, _semantic=_semantic)
    rem = coord.__mod__(align_val, _semantic=_semantic)
    is_zero = rem.__eq__(zero, _semantic=_semantic)

    fp4_padded = "with fp4_padded=True " if desc.layout.fp4_padded else ""
    ttgl.device_assert(
        is_zero, f"{fn_name} {fp4_padded}{arg_name} must be {align_bytes}-byte aligned, "
        f"i.e. a multiple of {align} for dtype={dtype.codegen_name()}", _semantic=_semantic)


def _convert_im2col_offsets(offsets, _semantic):
    offsets_ir = []
    for offset in offsets:
        offset = _unwrap_if_constexpr(offset)
        if isinstance(offset, int):
            offsets_ir.append(_semantic.builder.get_int16(offset))
        elif hasattr(offset, "handle"):
            offsets_ir.append(offset.handle)
        else:
            raise ValueError(f"Unsupported offset type: {type(offset)}")
    return offsets_ir


@builtin
def async_load(tensor_desc, coord, barrier, result, pred=True, multicast=False, _semantic=None):
    """
    Load data from global memory to shared memory using TMA.

    Args:
        tensor_desc: Tensor descriptor (tiled)
        coord: Coordinates in the source tensor
        barrier: Barrier for synchronization. In a two-CTA kernel, use a
            two-CTA barrier when this TMA load feeds a tcgen05 op; otherwise
            use a barrier allocated with ``two_ctas=False``.
        result: Destination memory descriptor
        pred: Predicate for conditional execution
        multicast: Enable multicast
    """
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, coord, "async_load", "innermost coordinate", _semantic=_semantic)

    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    pred = _semantic.to_tensor(pred)
    multicast = _unwrap_if_constexpr(multicast)

    _semantic.builder.create_async_tma_copy_global_to_local(
        tensor_desc.handle,
        coord,
        barrier.handle,
        result.handle,
        pred.handle,
        multicast,
        None,
    )


@builtin
def async_load_im2col(tensor_desc, coord, offsets, barrier, result, pred=True, multicast=False, _semantic=None):
    """
    Load data from global memory to shared memory using TMA in im2col mode.

    Args:
        tensor_desc: Tensor descriptor (im2col)
        coord: Coordinates in the source tensor
        offsets: Im2col offsets (must be i16 values)
            - For 3D tensors: 1 offset
            - For 4D tensors: 2 offsets
            - For 5D tensors: 3 offsets
        barrier: Barrier for synchronization. In a two-CTA kernel, use a
            two-CTA barrier when this TMA load feeds a tcgen05 op; otherwise
            use a barrier allocated with ``two_ctas=False``.
        result: Destination memory descriptor
        pred: Predicate for conditional execution
        multicast: Enable multicast
    """
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, coord, "async_load", "innermost coordinate", _semantic=_semantic)

    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    pred = _semantic.to_tensor(pred)
    multicast = _unwrap_if_constexpr(multicast)
    offsets_ir = _convert_im2col_offsets(offsets, _semantic)

    _semantic.builder.create_async_tma_copy_global_to_local(
        tensor_desc.handle,
        coord,
        barrier.handle,
        result.handle,
        pred.handle,
        multicast,
        offsets_ir,
    )


@builtin
def async_store(tensor_desc, coord, src, _semantic=None):
    """
    Store data from shared memory to global memory using TMA.

    Args:
        tensor_desc (tensor_descriptor): Tensor descriptor (tiled).
        coord (Sequence[int | ttgl.constexpr | ttgl.tensor]): Coordinates in the destination tensor.
        src (ttgl.shared_memory_descriptor): Source memory descriptor.
    """
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, coord, "async_store", "innermost coordinate", _semantic=_semantic)
    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    _semantic.builder.create_async_tma_copy_local_to_global(tensor_desc.handle, coord, src.handle)


# Backward-compatible aliases
async_copy_global_to_shared = async_load
async_copy_global_to_shared_im2col = async_load_im2col
async_copy_shared_to_global = async_store


def _async_atomic_shared_to_global(kind, tensor_desc, coord, src, fn_name: str, _semantic=None):
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, coord, fn_name, "innermost coordinate", _semantic=_semantic)
    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    _semantic.builder.create_async_tma_reduce(kind, tensor_desc.handle, coord, src.handle)


@builtin
def async_atomic_add(tensor_desc, coord, src, _semantic=None):
    """
    Atomically add data from shared memory into global memory using TMA.

    Args:
        tensor_desc (tensor_descriptor): Tensor descriptor (tiled).
        coord (Sequence[int | ttgl.constexpr | ttgl.tensor]): Coordinates in the destination tensor.
        src (ttgl.shared_memory_descriptor): Source memory descriptor.
    """
    _async_atomic_shared_to_global(ttgl.ir.DESCRIPTOR_REDUCE_KIND.ADD, tensor_desc, coord, src, "async_atomic_add",
                                   _semantic=_semantic)


@builtin
def async_atomic_min(tensor_desc, coord, src, _semantic=None):
    """
    Atomically compute the minimum of shared memory data and global memory using TMA.

    Args:
        tensor_desc (tensor_descriptor): Tensor descriptor (tiled).
        coord (Sequence[int | ttgl.constexpr | ttgl.tensor]): Coordinates in the destination tensor.
        src (ttgl.shared_memory_descriptor): Source memory descriptor.
    """
    _async_atomic_shared_to_global(ttgl.ir.DESCRIPTOR_REDUCE_KIND.MIN, tensor_desc, coord, src, "async_atomic_min",
                                   _semantic=_semantic)


@builtin
def async_atomic_max(tensor_desc, coord, src, _semantic=None):
    """
    Atomically compute the maximum of shared memory data and global memory using TMA.

    Args:
        tensor_desc (tensor_descriptor): Tensor descriptor (tiled).
        coord (Sequence[int | ttgl.constexpr | ttgl.tensor]): Coordinates in the destination tensor.
        src (ttgl.shared_memory_descriptor): Source memory descriptor.
    """
    _async_atomic_shared_to_global(ttgl.ir.DESCRIPTOR_REDUCE_KIND.MAX, tensor_desc, coord, src, "async_atomic_max",
                                   _semantic=_semantic)


@builtin
def async_atomic_and(tensor_desc, coord, src, _semantic=None):
    """
    Atomically bitwise-and data from shared memory into global memory using TMA.

    Args:
        tensor_desc (tensor_descriptor): Tensor descriptor (tiled).
        coord (Sequence[int | ttgl.constexpr | ttgl.tensor]): Coordinates in the destination tensor.
        src (ttgl.shared_memory_descriptor): Source memory descriptor.
    """
    _async_atomic_shared_to_global(ttgl.ir.DESCRIPTOR_REDUCE_KIND.AND, tensor_desc, coord, src, "async_atomic_and",
                                   _semantic=_semantic)


@builtin
def async_atomic_or(tensor_desc, coord, src, _semantic=None):
    """
    Atomically bitwise-or data from shared memory into global memory using TMA.

    Args:
        tensor_desc (tensor_descriptor): Tensor descriptor (tiled).
        coord (Sequence[int | ttgl.constexpr | ttgl.tensor]): Coordinates in the destination tensor.
        src (ttgl.shared_memory_descriptor): Source memory descriptor.
    """
    _async_atomic_shared_to_global(ttgl.ir.DESCRIPTOR_REDUCE_KIND.OR, tensor_desc, coord, src, "async_atomic_or",
                                   _semantic=_semantic)


@builtin
def async_atomic_xor(tensor_desc, coord, src, _semantic=None):
    """
    Atomically bitwise-xor data from shared memory into global memory using TMA.

    Args:
        tensor_desc (tensor_descriptor): Tensor descriptor (tiled).
        coord (Sequence[int | ttgl.constexpr | ttgl.tensor]): Coordinates in the destination tensor.
        src (ttgl.shared_memory_descriptor): Source memory descriptor.
    """
    _async_atomic_shared_to_global(ttgl.ir.DESCRIPTOR_REDUCE_KIND.XOR, tensor_desc, coord, src, "async_atomic_xor",
                                   _semantic=_semantic)


@builtin
def store_wait(pendings, read_only=True, _semantic=None):
    """
    Wait for pending TMA stores.

    Args:
        pendings (int | ttgl.constexpr): Maximum number of TMA stores allowed to remain pending.
        read_only (bool | ttgl.constexpr): If true, wait only until the pending stores have finished reading
            their shared-memory sources, but writes may not be visible in HBM. Defaults to true.

    Notes:
        By default, ``tma.store_wait`` only waits for the TMA store to finish reading from the shared memory,
        however this does not mean that the write has been fully flushed to HBM. If your kernel uses TMA to pass
        messages between CTAs, or between nvlink devices then you will need to use ``read_only=False`` before any
        release operation.
    """
    pendings = _unwrap_if_constexpr(pendings)
    read_only = _unwrap_if_constexpr(read_only)
    _semantic.builder.create_async_tma_store_wait(pendings, read_only)


@builtin
def make_tensor_descriptor(
    base: ttgl.tensor,
    shape: List[ttgl.tensor],
    strides: List[ttgl.tensor],
    block_shape: List[ttgl.constexpr],
    layout: NVMMASharedLayout,
    padding_option="zero",
    _semantic=None,
) -> tensor_descriptor:
    padding_option = _unwrap_if_constexpr(padding_option)
    block_shape = _unwrap_if_constexpr(block_shape)

    ndim = len(shape)
    if not (1 <= ndim <= 5):
        raise ValueError(f"Expected 1 <= ndim <= 5 but got {ndim} dimensions")
    if len(strides) != ndim:
        raise ValueError(f"Expected {ndim} strides but got {len(strides)}")
    if len(block_shape) != ndim:
        raise ValueError(f"Expected block_shape to have {ndim} dimensions but got {len(block_shape)}")
    assert isinstance(base.dtype, ttgl.pointer_type)
    elem_size = base.dtype.element_ty.primitive_bitwidth // 8
    contig_dim_size = ttgl._unwrap_if_constexpr(block_shape[-1])
    if contig_dim_size * elem_size < 16:
        raise ValueError(
            f"Descriptor block shape must have at least 16 bytes in the last dimension, but got {contig_dim_size} * {elem_size} = {contig_dim_size * elem_size} bytes"
        )

    last_stride = ttgl._unwrap_if_constexpr(strides[-1])
    if last_stride != 1:
        raise ValueError(f"Tensor descriptor last dim must be 1 but got {last_stride}")

    shape = [_semantic.make_scalar(x, ttgl.int32) for x in shape]
    strides = [_semantic.make_scalar(ttgl._unwrap_if_constexpr(x), ttgl.int64) for x in strides]

    # Check whether `block_shape` is static
    block_shape = ttgl._unwrap_shape(block_shape)

    assert isinstance(base.type, ttgl.pointer_type)
    block_type = ttgl.block_type(base.type.element_ty, block_shape)
    base_handle = base.handle

    padding = _semantic._str_to_padding_option(padding_option)

    layout = _unwrap_if_constexpr(layout)
    assert isinstance(layout, NVMMASharedLayout), \
        "Expected layout to be a NVMMASharedLayout"

    shape_type = ttgl.tuple(shape).type
    strides_type = ttgl.tuple(strides).type
    ty = tensor_descriptor_type(block_type, shape_type, strides_type, layout)

    if base.type.element_ty.is_int() and padding == ttgl.ir.PADDING_OPTION.PAD_NAN:
        raise ValueError("Padding option `nan` is not supported for integer blocks")
    handle = _semantic.builder.create_make_tensor_descriptor(
        ty._to_ir(_semantic.builder),
        base_handle,
        [s.handle for s in shape],
        [s.handle for s in strides],
        padding,
    )
    return tensor_descriptor(handle, shape, strides, block_type, layout)
<|endoftext|>from ..ampere.mbarrier import MBarrierLayout, allocate_mbarrier, init, invalidate, wait
from ..._core import _unwrap_if_constexpr, builtin

__all__ = [
    "allocate_mbarrier",
    "arrive",
    "expect",
    "fence_init_release_cluster",
    "init",
    "invalidate",
    "MBarrierLayout",
    "wait",
]


@builtin
def expect(mbarrier, bytes_per_cta=None, pred=True, _semantic=None):
    """
    Expect a specific number of bytes being copied. When they are copied, the barrier is signaled.

    Args:
        mbarrier (shared_memory_descriptor): Barrier that will be signaled when the operation is complete.
        bytes_per_cta (int): Expected byte count per CTA.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    pred = _semantic.to_tensor(pred)
    bytes_per_cta = _unwrap_if_constexpr(bytes_per_cta)
    _semantic.builder.create_mbarrier_expect(mbarrier.handle, bytes_per_cta, pred.handle)


@builtin
def arrive(mbarrier, *, count=1, pred=True, _semantic=None):
    """
    Arrive at an mbarrier with a specified count.

    Args:
        mbarrier (shared_memory_descriptor): Barrier to be signalled.
        count (int): Count to arrive with. Defaults to 1.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    count = _unwrap_if_constexpr(count)
    cta_mask = 0
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, cta_mask, pred.handle)


@builtin
def fence_init_release_cluster(_semantic=None):
    """
    Fence that makes prior mbarrier initialization visible across the CTA cluster.

    Needs to be called together with cluster.barrier(relaxed=True).
    """
    _semantic.builder.create_fence_mbarrier_init_release_cluster()
<|endoftext|>"""
Cluster Launch Control (CLC) for Blackwell (SM100+) dynamic persistent kernels.

CLC enables hardware-based dynamic work scheduling where running workers can
cancel not-yet-launched clusters and take over their work via the
clusterlaunchcontrol.try_cancel instruction.
"""
from __future__ import annotations

import triton.experimental.gluon.language._core as gl
from triton.experimental.gluon.language._core import builtin, tensor, shared_memory_descriptor, base_value, base_type
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
    from triton._C.libtriton import ir

__all__ = [
    "try_cancel",
    "load_result",
    "clc_result",
]


@builtin
def try_cancel(result: shared_memory_descriptor, barrier, _semantic=None):
    """
    Issue a CLC try_cancel request to atomically cancel a pending cluster launch.

    Args:
        result (shared_memory_descriptor): 16-byte aligned int64x2 shared memory for the response
        barrier (shared_memory_descriptor): 8-byte aligned mbarrier for completion signaling

    Only supported on SM100+ (Blackwell).
    """
    _semantic.builder.create_clc_try_cancel(result.handle, barrier.handle)


@builtin
def load_result(src, _semantic=None):
    """
    Load the CLC response from shared memory into registers.

    Args:
        src (shared_memory_descriptor): The int64x2 CLC response buffer

    Returns:
        CLCResult: Object with is_canceled() and get_first_ctaid(dim) methods
    """
    handle = _semantic.builder.create_clc_load_result(src.handle)
    return clc_result(handle)


class clc_result_type(base_type):

    def to_ir(self, builder: GluonOpBuilder) -> None:
        return builder.get_int128_ty()

    def _unflatten_ir(self, handles: List[ir.Value], cursor: int) -> Tuple[shared_memory_descriptor, int]:
        value = clc_result(handles[cursor])
        return value, cursor + 1

    def _flatten_ir_types(self, builder: GluonOpBuilder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def __str__(self) -> str:
        return "clc_result"

    def __eq__(self, other) -> bool:
        return type(self) is type(other)

    def mangle(self) -> str:
        return "CLC"


class clc_result(base_value):
    """CLC response loaded into registers. Query without re-reading memory."""

    def __init__(self, handle):
        self.handle = handle
        self.type = clc_result_type()

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))

    @builtin
    def is_canceled(self, _semantic=None):
        """
        Check if the CLC response indicates a successful cancellation.

        Returns:
            tensor: True if a cluster was successfully canceled, False otherwise
        """
        handle = _semantic.builder.create_clc_is_canceled(self.handle)
        return tensor(handle, gl.int1)

    @builtin
    def program_id(self, dim, _semantic=None):
        """
        Get the Program ID of the canceled cluster.

        Args:
            dim (int): Dimension to get (0=x, 1=y, 2=z)

        Returns:
            tensor: The Program ID for the specified dimension
        """
        handle = _semantic.builder.create_clc_get_program_id(self.handle, dim)
        return tensor(handle, gl.int32)
<|endoftext|>from __future__ import annotations
from typing import Tuple, List, TYPE_CHECKING

from dataclasses import dataclass, field
from triton.runtime.jit import constexpr_function
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._core import builtin, base_type, base_value, _unwrap_if_constexpr
from triton.experimental.gluon.language._semantic import _check, _compute_tmem_reg_layout

from . import tma
from . import clc
from ..hopper import async_store, fence_async_shared, mbarrier
from ..ampere import async_copy, mma_v2

from triton._C.libtriton import ir
import triton._C.libtriton.gluon_ir as gluon_ir
if TYPE_CHECKING:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
    from ..._semantic import GluonSemantic

__all__ = [
    "add2",
    "allocate_tensor_memory",
    "async_copy",
    "async_store",
    "clc",
    "fence_async_shared",
    "fma2",
    "mbarrier",
    "max2",
    "min2",
    "mma_v2",
    "mul2",
    "sub2",
    "tensor_memory_descriptor",
    "tensor_memory_descriptor_type",
    "TensorMemoryLayout",
    "TensorMemoryScalesLayout",
    "tma",
    "_TensorMemoryLinearLayout",
]


def _packed_arith(operation, operands, dtype, semantic):
    """Build packed arithmetic with inferred result types and FP4 layouts."""
    operands = tuple(semantic.to_tensor(operand) for operand in operands)
    _check(any(isinstance(operand.type, ttgl.distributed_type) for operand in operands),
           lambda: "packed arithmetic requires at least one distributed tensor operand")
    floating = [
        operand for operand in operands
        if isinstance(operand.type, ttgl.distributed_type) and operand.dtype.is_floating()
    ]
    reference = max(floating, key=lambda operand: operand.numel.value, default=None)
    dtype = _unwrap_if_constexpr(dtype)

    if reference is None:
        _check(dtype is not None, lambda: "packed FP4 operands require an explicit result dtype")
        shape = list(operands[0].type.shape)
        shape[-1] *= 2
    else:
        shape = reference.type.shape
        if dtype is None:
            addend = operands[-1]
            dtype = addend.dtype if operation in ("add", "sub", "fma") and addend.dtype.is_fp8() else reference.dtype
            for operand in operands:
                if not operand.dtype.is_floating() or (dtype.is_fp8() and operand.dtype.is_fp8()):
                    continue
                dtype = semantic.computation_type_impl(dtype, False, operand.dtype, not operand.type.is_block(), False)

    _check(isinstance(dtype, ttgl.dtype), lambda: f"expected 'dtype' to be a dtype but got {dtype}")
    normalized = []
    for operand in operands:
        is_packed_fp4 = (isinstance(operand.type, ttgl.distributed_type) and operand.dtype.is_int()
                         and operand.dtype.primitive_bitwidth == 8 and operand.type.shape != shape)
        if not is_packed_fp4:
            _check(reference is not None, lambda: "packed FP4 operands require a floating-point tensor operand")
            if not operand.type.is_block():
                operand = semantic.cast(operand, dtype)
            _, operand = semantic.broadcast_impl_value(reference, operand)
        normalized.append(operand.handle)

    builder = semantic.builder
    result_type = dtype if reference is None else reference.type.with_element_ty(dtype)
    result_ir_type = result_type.to_ir(builder)
    handle = builder.create_packed_arith(result_ir_type, operation, normalized)
    return semantic._wrap_handle_infer_layout(handle, dtype, shape)


@builtin
def add2(lhs, rhs, dtype=None, _semantic=None):
    """Add two tensors using a native two-lane packed instruction."""
    return _packed_arith("add", (lhs, rhs), dtype, _semantic)


@builtin
def sub2(lhs, rhs, dtype=None, _semantic=None):
    """Subtract two tensors using a native two-lane packed instruction."""
    return _packed_arith("sub", (lhs, rhs), dtype, _semantic)


@builtin
def mul2(lhs, rhs, dtype=None, _semantic=None):
    """Multiply two tensors using a native two-lane packed instruction."""
    return _packed_arith("mul", (lhs, rhs), dtype, _semantic)


@builtin
def fma2(lhs, rhs, acc, dtype=None, _semantic=None):
    """Perform a native two-lane packed fused multiply-add."""
    return _packed_arith("fma", (lhs, rhs, acc), dtype, _semantic)


@builtin
def min2(lhs, rhs, dtype=None, _semantic=None):
    """Select the minimum with a native packed half-precision instruction."""
    return _packed_arith("min", (lhs, rhs), dtype, _semantic)


@builtin
def max2(lhs, rhs, dtype=None, _semantic=None):
    """Select the maximum with a native packed half-precision instruction."""
    return _packed_arith("max", (lhs, rhs), dtype, _semantic)


@dataclass(frozen=True, eq=True)
class TensorMemoryLayout:
    """
    Describes the layout for tensor memory in Blackwell architecture.

    Args:
        block (Tuple[int, int]): Number of contiguous elements per row / column in a CTA.
        col_stride (int): Number of 32-bit columns to advance between logically
            adjacent columns. Packed layouts use a stride of 1. Unpacked
            layouts use ``32 / bitwidth``.
        cga_layout (Optional[List[List[int]]]): CGA layout bases. Defaults to [].
        two_ctas (bool): Whether the layout is for two-CTA mode. Defaults to False.
        fp4_padded (bool): Whether byte-backed operand A uses the padded MMAv5
            FP4 layout. Its descriptor keeps the packed ``Mx(K/2)xi8`` shape,
            MMAv5 treats logical K as twice descriptor K, and physical TMEM
            reserves one byte per logical FP4 element. Defaults to False.
    """
    block: Tuple[int, int]
    col_stride: int
    cga_layout: List[List[int]] = field(default_factory=list)
    two_ctas: bool = False
    fp4_padded: bool = False

    def __post_init__(self):
        super().__setattr__("block", _unwrap_if_constexpr(self.block))
        super().__setattr__("col_stride", _unwrap_if_constexpr(self.col_stride))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))
        super().__setattr__("two_ctas", _unwrap_if_constexpr(self.two_ctas))
        super().__setattr__("fp4_padded", _unwrap_if_constexpr(self.fp4_padded))
        assert len(self.block) == 2
        assert all(len(basis) == 2 for basis in self.cga_layout)
        assert self.col_stride >= 1 and (self.col_stride &
                                         (self.col_stride - 1)) == 0, "tensor memory col_stride must be a power of two"

    def _to_ir(self, builder):
        return builder.get_tensor_memory_layout(
            self.block,
            self.col_stride,
            [list(basis) for basis in self.cga_layout],
            self.two_ctas,
            self.fp4_padded,
        )

    def mangle(self) -> str:
        block_str = f"{self.block[0]}x{self.block[1]}"
        stride_str = f"C{self.col_stride}"
        cga_layout_str = "_".join("~".join(map(str, basis)) for basis in self.cga_layout)
        two_ctas_str = "2CT" if self.two_ctas else ""
        fp4_padded_str = "FP4P" if self.fp4_padded else ""
        return f"TL{block_str}{stride_str}{cga_layout_str}{two_ctas_str}{fp4_padded_str}TL"

    def __hash__(self):
        return hash(
            (tuple(self.block), self.col_stride, tuple(tuple(b)
                                                       for b in self.cga_layout), self.two_ctas, self.fp4_padded))


@dataclass(frozen=True, eq=True)
class TensorMemoryScalesLayout:
    """
    Describes the layout for tensor memory scales in Blackwell architecture.

    Args:
        cga_layout (Optional[List[List[int]]]): CGA layout bases. Defaults to [].
    """
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))
        assert all(len(basis) == 2 for basis in self.cga_layout)

    def _to_ir(self, builder):
        return builder.get_tensor_memory_scales_layout([list(basis) for basis in self.cga_layout])

    def mangle(self) -> str:
        cga_layout_str = "_".join("~".join(map(str, basis)) for basis in self.cga_layout)
        return f"TLS{cga_layout_str}TLS"

    def __hash__(self):
        return hash(tuple(tuple(b) for b in self.cga_layout))


@dataclass(frozen=True)
class _TensorMemoryLinearLayout:
    """
    Print-only linear layout for TMEM (row/col -> dim0/dim1).
    """
    rows: List[List[int]]
    cols: List[List[int]]
    shape: List[int]

    def _to_ir(self, builder):
        raise RuntimeError("TensorMemoryLinearLayout is print-only; IR materialization is unsupported")

    def mangle(self):
        return f"TMLL_{self.shape}_TMLL"

    def __hash__(self):
        return hash((tuple(map(tuple, self.rows)), tuple(map(tuple, self.cols)), tuple(self.shape)))


def _unwrap_tmem_layout_arg(x):
    if isinstance(x, ttgl.constexpr):
        return _unwrap_tmem_layout_arg(x.value)
    if isinstance(x, list):
        return [_unwrap_tmem_layout_arg(i) for i in x]
    if isinstance(x, tuple):
        return tuple(_unwrap_tmem_layout_arg(i) for i in x)
    return x


class tensor_memory_descriptor_type(base_type):

    def __init__(self, element_ty, shape, layout, alloc_shape):
        self.element_ty = _unwrap_if_constexpr(element_ty)
        self.shape = _unwrap_if_constexpr(shape)
        self.layout = _unwrap_if_constexpr(layout)
        self.alloc_shape = _unwrap_if_constexpr(alloc_shape)
        assert isinstance(self.layout, (TensorMemoryLayout, TensorMemoryScalesLayout))

    def to_ir(self, builder: GluonOpBuilder) -> None:
        return builder.get_tensor_mem_desc_ty(
            self.element_ty.to_ir(builder),
            self.shape,
            self.layout._to_ir(builder),
            self.alloc_shape,
        )

    def _unflatten_ir(self, handles: List[ir.Value], cursor: int) -> Tuple[tensor_memory_descriptor, int]:
        value = tensor_memory_descriptor(handles[cursor], self.element_ty, self.shape, self.layout, self.alloc_shape)
        return value, cursor + 1

    def _flatten_ir_types(self, builder: GluonOpBuilder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def __str__(self) -> str:
        return f"tensor_memory_descriptor<{self.element_ty}, {self.shape}, {self.layout}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.layout == other.layout
                and self.alloc_shape == other.alloc_shape)

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        shape_str = "_".join([str(s) for s in self.shape])
        return f"MD{self.element_ty.mangle()}S{shape_str}SL{self.layout.mangle()}LAS{self.alloc_shape}ASMD"

    @constexpr_function
    def get_reg_layout(self, num_warps=None, instr_variant="32x32b"):
        """
        Return a DistributedLinearLayout compatible with TMEM load/store
        instructions for this descriptor type.

        Args:
            num_warps (Optional[int]): Number of warps participating in the
                operation. Must be provided when it cannot be inferred by the
                caller.
            instr_variant (str): TMEM instruction variant (e.g. ``"32x32b"``).
        """
        tmem_ty = _unwrap_tmem_layout_arg(self)
        if not isinstance(tmem_ty, tensor_memory_descriptor_type):
            raise TypeError(f"expected a tensor_memory_descriptor_type but got {type(tmem_ty)!r}")
        num_warps = _unwrap_tmem_layout_arg(num_warps)
        if num_warps is None:
            raise ValueError("num_warps could not be inferred; pass a positive power of two")
        if not isinstance(num_warps, int) or num_warps <= 0 or (num_warps & (num_warps - 1)) != 0:
            raise ValueError(f"num_warps must be a positive power of two, got {num_warps!r}")

        return _compute_tmem_reg_layout(
            _unwrap_tmem_layout_arg(tmem_ty.element_ty),
            _unwrap_tmem_layout_arg(tmem_ty.shape),
            _unwrap_tmem_layout_arg(tmem_ty.alloc_shape),
            _unwrap_tmem_layout_arg(tmem_ty.layout),
            num_warps,
            _unwrap_tmem_layout_arg(instr_variant),
        )


class tensor_memory_descriptor(base_value):
    """
    Represents a tensor memory descriptor handle for Tensor Core Gen5 operations.
    """

    def __init__(self, handle, element_ty, shape, layout, alloc_shape):
        self.handle = handle
        self.type = tensor_memory_descriptor_type(element_ty, shape, layout, alloc_shape)

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def dtype(self):
        return self.type.element_ty

    @property
    def shape(self):
        return self.type.shape

    @property
    def rank(self):
        return len(self.shape)

    @property
    def layout(self):
        return self.type.layout

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def get_reg_layout(self, num_warps=None, instr_variant="32x32b", _semantic: GluonSemantic = None, _generator=None):
        """
        Return the register layout used to access this tensor memory descriptor.

        Args:
            num_warps (Optional[int]): Number of warps participating in the
                operation. When omitted, infer it from the caller's context.
            instr_variant (str): TMEM instruction variant. Defaults to
                ``"32x32b"``.

        Returns:
            DistributedLayout: A register layout compatible with TMEM
            load/store instructions for this descriptor.
        """
        if num_warps is None:
            num_warps = ttgl.num_warps(_semantic=_semantic, _generator=_generator)
        return self.type.get_reg_layout(num_warps=num_warps, instr_variant=instr_variant)

    @builtin
    def load(self, layout=None, _semantic: GluonSemantic = None, _generator=None) -> ttgl.tensor:
        """
        Load a tensor from tensor memory.

        Args:
            layout (Optional[DistributedLayout]): Destination layout of the tensor.
                When omitted, infer the default TMEM register layout for this
                descriptor type and the caller's warp count.

        Returns:
            tensor: A distributed tensor containing the loaded data.
        """
        if layout is None:
            layout = self.get_reg_layout(_semantic=_semantic, _generator=_generator)
        layout = _unwrap_if_constexpr(layout)
        ret_ty = ttgl.distributed_type(self.dtype, self.shape, layout)
        builder = _semantic.builder
        handle = builder.create_tmem_load(ret_ty.to_ir(builder), self.handle)
        return ttgl.tensor(handle, ret_ty)

    def _load_red(self, layout, red_op, abs, propagate_nan, _semantic: GluonSemantic, _generator=None):
        #   red_op: MIN/MAX reduction operation
        #   abs (bool): If True, reduce absolute values.
        #   propagate_nan (NONE): If ALL, propagate NaN in specified reduction operation.
        if layout is None:
            layout = self.get_reg_layout(_semantic=_semantic, _generator=_generator)
        layout = _unwrap_if_constexpr(layout)
        abs_flag = _unwrap_if_constexpr(abs)
        propagate_nan = _unwrap_if_constexpr(propagate_nan)

        ret_ty = ttgl.distributed_type(self.dtype, self.shape, layout)
        builder = _semantic.builder

        result, reduced, red_layout = builder.create_tmem_load(ret_ty.to_ir(builder), self.handle, red_op, abs_flag,
                                                               propagate_nan)

        red_shape = [self.shape[0]]  # [M] for [M,N] input
        red_ty = ttgl.distributed_type(self.dtype, red_shape, red_layout)

        return (ttgl.tensor(result, ret_ty), ttgl.tensor(reduced, red_ty))

    @builtin
    def load_min(self, layout=None, abs=False, propagate_nan=ir.PROPAGATE_NAN.NONE, _semantic: GluonSemantic = None,
                 _generator=None):
        """
        Load a tensor from tensor memory with MIN reduction along the N-dimension.

        Args:
            layout (Optional[DistributedLayout]): Destination layout of the tensor.
                When omitted, infer the default TMEM register layout for this
                descriptor type and the caller's warp count.
            abs (bool): If True, reduce absolute values. Defaults to False.
            propagate_nan (PROPAGATE_NAN): If ALL, propagate NaN in the reduction operation. Defaults to NONE.

        Returns:
            tuple: A tuple containing (tensor, reduced_tensor) where tensor is the loaded data
                   and reduced_tensor is the result of MIN reduction along the N-dimension of loaded data
        """
        return self._load_red(layout, gluon_ir.TMEM_LOAD_REDUCE_MODIFIER.MIN, abs, propagate_nan, _semantic, _generator)

    @builtin
    def load_max(self, layout=None, abs=False, propagate_nan=ir.PROPAGATE_NAN.NONE, _semantic: GluonSemantic = None,
                 _generator=None):
        """
        Load a tensor from tensor memory with MAX reduction along the N-dimension.

        Args:
            layout (Optional[DistributedLayout]): Destination layout of the tensor.
                When omitted, infer the default TMEM register layout for this
                descriptor type and the caller's warp count.
            abs (bool): If True, reduce absolute values. Defaults to False.
            propagate_nan (PROPAGATE_NAN): If ALL, propagate NaN in the reduction operation. Defaults to NONE.

        Returns:
            tuple: A tuple containing (tensor, reduced_tensor) where tensor is the loaded data
                   and reduced_tensor is the result of MAX reduction along the N-dimension of loaded data.
        """
        return self._load_red(layout, gluon_ir.TMEM_LOAD_REDUCE_MODIFIER.MAX, abs, propagate_nan, _semantic, _generator)

    @builtin
    def store(self, value, pred=True, _semantic: GluonSemantic = None) -> None:
        """
        Store a tensor into tensor memory.

        Args:
            value (tensor): The tensor to store.
            pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        """
        pred = _unwrap_if_constexpr(pred)
        pred = _semantic.to_tensor(pred)
        assert value.shape == self.shape, f"source shape {value.shape} does not match destination shape {self.shape}"
        assert value.dtype == self.dtype, f"source dtype {value.dtype} does not match destination dtype {self.dtype}"
        _semantic.builder.create_tmem_store(self.handle, value.handle, pred.handle)

    @builtin
    def slice(self, start, length, _semantic: GluonSemantic = None) -> None:
        """
        Create a slice of the tensor memory descriptor along the last dimension.

        Args:
            start (int): The starting index for subslice.
            length (int): The length of the subslice.

        Returns:
            tensor_memory_descriptor: Descriptor for the subslice.
        """
        start = _unwrap_if_constexpr(start)
        length = _unwrap_if_constexpr(length)
        _check(isinstance(start, int), lambda: "start must be a constant int")
        _check(isinstance(length, int), lambda: "length must be a constant int")
        shape = self.shape[:-1] + [length]
        layout = self.type.layout
        ret = tensor_memory_descriptor(None, self.dtype, shape, layout, self.type.alloc_shape)
        builder = _semantic.builder
        ret.handle = builder.create_tmem_subslice(ret.type.to_ir(builder), self.handle, start)
        return ret

    @builtin
    def index(self, index, _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        """
        Create a subview of tensor memory by indexing the first dimension.

        Args:
            index (tensor): The index tensor for the subview.

        Returns:
            tensor_memory_descriptor: Descriptor for the indexed subview.
        """
        index = _semantic.to_tensor(index)
        builder = _semantic.builder
        shape = self.shape[1:]
        layout = self.layout
        ret = tensor_memory_descriptor(None, self.dtype, shape, layout, shape)
        ret.handle = builder.create_memdesc_index(ret.type.to_ir(builder), self.handle, index.handle)
        return ret

    @builtin
    def _reinterpret(self, dtype=None, shape=None, layout=None,
                     _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        """
        Reinterpret tensor memory descriptor with a new dtype, shape, and layout.

        Args:
            dtype (dtype): The new data type. Defaults to the descriptor dtype.
            shape (Sequence[int]): The new shape. Defaults to the descriptor shape.
            layout (TensorMemoryLayout): The new layout. Defaults to the descriptor layout.

        Returns:
            tensor_memory_descriptor: Descriptor with updated type and layout.
        """
        dtype = self.dtype if dtype is None else _unwrap_if_constexpr(dtype)
        shape = self.shape if shape is None else [_unwrap_if_constexpr(s) for s in shape]
        layout = self.layout if layout is None else _unwrap_if_constexpr(layout)

        ty = tensor_memory_descriptor_type(dtype, shape, layout, shape)
        handle = _semantic.builder.create_memdesc_reinterpret(ty.to_ir(_semantic.builder), self.handle)
        return tensor_memory_descriptor(handle, **ty.__dict__)


@builtin
def allocate_tensor_memory(element_ty, shape, layout, value=None, _semantic=None):
    """
    Allocate tensor memory.

    Args:
        element_ty (dtype): The element data type.
        shape (Sequence[int]): The descriptor shape.
        layout (TensorMemoryLayout): The layout of the tensor memory.
        value (tensor, optional): Initial tensor to copy. Defaults to None.

    Returns:
        tensor_memory_descriptor: Descriptor for the allocated memory.
    """
    element_ty = _unwrap_if_constexpr(element_ty)
    shape = _unwrap_if_constexpr(shape)
    layout = _unwrap_if_constexpr(layout)
    value = value.handle if value is not None else None

    ty = tensor_memory_descriptor_type(element_ty, shape, layout, shape)
    builder = _semantic.builder
    handle = builder.create_tmem_alloc(ty.to_ir(builder), value)
    return tensor_memory_descriptor(handle, element_ty, shape, layout, shape)


@builtin
def tcgen05_copy(src, dst, _semantic=None):
    """
    Start an asynchronous copy from shared memory to tensor memory.

    Args:
        src (shared_memory_descriptor): Shared memory to copy from.
        dst (tensor_memory_descriptor): Tensor memory to copy to.
    """
    assert isinstance(src, ttgl.shared_memory_descriptor), "source must be a shared memory descriptor"
    assert isinstance(dst, tensor_memory_descriptor), "destination must be a tensor memory descriptor"
    _semantic.builder.create_tmem_copy(src.handle, dst.handle)


@builtin
def tcgen05_mma(a, b, acc, *, use_acc=True, pred=True, multicast=False, mbarriers=None, mbarrier_preds=None,
                _semantic=None):
    """
    Emit a 5th generation TensorCore MMA instruction.
    acc = a * b + (acc if use_acc else 0)

    Args:
        a (shared_memory_descriptor or tensor_memory_descriptor): Left hand side operand in shared or tensor memory.
        b (shared_memory_descriptor): Right hand side operand in shared memory.
        acc (tensor_memory_descriptor): Accumulator value in tensor memory (mutated).
        use_acc (bool): Whether to use the initial value of the accumulator. Defaults to True.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        multicast (bool): Whether tcgen05 commit should multicast across a CTA cluster. Defaults to False.
        mbarriers (Sequence[shared_memory_descriptor], optional): Barriers to signal when the operation is complete. If None, mma is synchronous. Defaults to None.
        mbarrier_preds (Sequence[bool], optional): Predicates for barriers. Defaults to None.
    """
    use_acc = _semantic.to_tensor(use_acc)
    pred = _semantic.to_tensor(pred)

    if mbarriers is None:
        assert mbarrier_preds is None
        mbarriers = []
        mbarrier_preds = []
    else:
        mbarriers = [bar.handle for bar in mbarriers]
        if mbarrier_preds is None:
            true = _semantic.to_tensor(True)
            mbarrier_preds = [true.handle] * len(mbarriers)
        else:
            mbarrier_preds = _semantic._convert_to_ir_values(mbarrier_preds, require_i64=False)

    multicast = _unwrap_if_constexpr(multicast)
    _semantic.builder.create_tcgen05_mma(a.handle, b.handle, acc.handle, use_acc.handle, pred.handle, mbarriers,
                                         mbarrier_preds, acc.layout.two_ctas, multicast)


@builtin
def tcgen05_mma_scaled(a, b, acc, a_scale, b_scale, a_type, b_type, *, use_acc=True, pred=True, multicast=False,
                       mbarriers=None, mbarrier_preds=None, _semantic=None):
    """
    Emit a 5th generation TensorCore MMA scaled instruction.
    acc = (a * a_scale) * (b * b_scale) + (acc if use_acc else 0)

    Args:
        a (shared_memory_descriptor): Left hand side operand in shared memory.
        b (shared_memory_descriptor or tensor_memory_descriptor): Right hand side operand in shared or tensor memory.
        acc (tensor_memory_descriptor): Accumulator value in tensor memory (mutated).
        a_scale (tensor): Scale factor for operand A.
        b_scale (tensor): Scale factor for operand B.
        a_type (str): Type of operand A. One of {"e2m1", "e4m3", "e5m2"}.
        b_type (str): Type of operand B. One of {"e2m1", "e4m3", "e5m2"}.
        use_acc (bool): Whether to use the initial value of the accumulator. Defaults to True.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        multicast (bool): Whether tcgen05 commit should multicast across a CTA cluster. Defaults to False.
        mbarriers (Sequence[mbarrier], optional): Barriers to signal when the operation is complete. If None, mma is synchronous. Defaults to None.
        mbarrier_preds (Sequence[bool], optional): Predicates for barriers. Defaults to None.
    """
    use_acc = _semantic.to_tensor(use_acc)
    pred = _semantic.to_tensor(pred)
    assert acc.type.layout.block[0] != 64, "tcgen05_mma_scaled does not support blockM=64"

    if mbarriers is None:
        assert mbarrier_preds is None
        mbarriers = []
        mbarrier_preds = []
    else:
        mbarriers = [bar.handle for bar in mbarriers]
        if mbarrier_preds is None:
            true = _semantic.to_tensor(True)
            mbarrier_preds = [true.handle] * len(mbarriers)
        else:
            mbarrier_preds = _semantic._convert_to_ir_values(mbarrier_preds, require_i64=False)

    allowed_formats = {"e2m1", "e4m3", "e5m2"}
    assert a_type.value in allowed_formats, f"Unsupported lhs_format: {a_type.value}"
    assert b_type.value in allowed_formats, f"Unsupported rhs_format: {b_type.value}"
    a_type = _semantic._str_to_fp_type(a_type.value)
    b_type = _semantic._str_to_fp_type(b_type.value)
    multicast = _unwrap_if_constexpr(multicast)
    _semantic.builder.create_tcgen05_mma_scaled(a.handle, b.handle, acc.handle, a_scale.handle, b_scale.handle, a_type,
                                                b_type, use_acc.handle, pred.handle, mbarriers, mbarrier_preds,
                                                acc.layout.two_ctas, multicast)


@constexpr_function
def tcgen05_mma_barrier_count(smems, multicast, two_ctas):
    """
    Calculate the number of CTAs that will commit the tcgen05 MMA instruction.

    Args:
        smems (Sequence[shared_memory_descriptor]): Shared memory descriptors used in the tcgen05 instruction.
        multicast (bool): Whether the tcgen05 instruction is multicast.
        two_ctas (bool): Whether the tcgen05 instruction uses cta_group::2.

    Returns:
        int: The number of CTAs that will commit the tcgen05 MMA instruction.
    """
    assert 0 <= len(smems) <= 4, "tcgen05_mma_barrier_count supports 0 to 4 descriptors"
    if not smems or not multicast:
        return 1

    def basis_is_zero(basis):
        return all(b == 0 for b in basis)

    num_cta_bits = len(smems[0].layout.cga_layout)
    for desc in smems[1:]:
        assert len(desc.layout.cga_layout) == num_cta_bits

    num_cta_commits = 0
    for cta in range(1 << num_cta_bits):
        if two_ctas and cta & 1:
            continue
        for desc in smems:
            if all(basis_is_zero(basis) or not (cta & (1 << i)) for i, basis in enumerate(desc.layout.cga_layout)):
                num_cta_commits += 1
                break
    return num_cta_commits


@builtin
def tcgen05_commit(barrier, pred=True, descs=(), _semantic=None):
    """
    This instruction causes the provided mbarrier to be arrived-on with a count
    of 1 when all async tcgen05 MMA and copy instructions previously issued by
    the thread are complete.

    If `descs` are provided, the commit will be multicast across the CTA cluster
    based on the shared layouts of those descriptors. This should be used when
    the inputs to the tcgen5 MMA come from TMA descriptors using multicast.

    Args:
        barrier (shared_memory_descriptor): The barrier to track completion of tcgen05 MMA and copy instructions.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        descs (Sequence[shared_memory_descriptor]): Shared memory descriptors for
            the preceding multiplication inputs. Defaults to ().
    """
    pred = _semantic.to_tensor(pred)
    descs = _unwrap_if_constexpr(descs)
    descs = [d.handle for d in descs]
    _semantic.builder.create_tcgen05_commit(barrier.handle, pred.handle, descs)
<|endoftext|>import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.nvidia.hopper.tma import (
    async_atomic_add,
    async_atomic_and,
    async_atomic_max,
    async_atomic_min,
    async_atomic_or,
    async_atomic_xor,
    async_copy_global_to_shared,
    async_copy_shared_to_global,
    async_load,
    async_load_im2col,
    async_store,
    store_wait,
    tensor_descriptor,
    tensor_descriptor_type,
    make_tensor_descriptor,
    _emit_alignment_check,
)

__all__ = [
    "async_gather",
    "async_scatter",
    "async_atomic_add",
    "async_atomic_and",
    "async_atomic_max",
    "async_atomic_min",
    "async_atomic_or",
    "async_atomic_xor",
    "async_copy_global_to_shared",
    "async_copy_shared_to_global",
    "async_load",
    "async_load_im2col",
    "async_store",
    "store_wait",
    "tensor_descriptor",
    "tensor_descriptor_type",
    "make_tensor_descriptor",
]


@builtin
def async_gather(tensor_desc, x_offsets, y_offset, barrier, result, pred=True, multicast=False, _semantic=None):
    """
    Asynchronously gather elements from global memory to shared memory using TMA.

    Args:
        tensor_desc (tensor_descriptor): The tensor descriptor.
        x_offsets (tensor): 1D tensor of X offsets.
        y_offset (int): Scalar Y offset.
        barrier (shared_memory_descriptor): Barrier that will be signaled when the operation is complete.
        result (tensor_memory_descriptor): Result shared memory, must have NVMMASharedLayout.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        multicast (bool): Enable multicast.
    """
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, (y_offset, ), "async_gather", "y_offset", _semantic=_semantic)

    pred = _semantic.to_tensor(pred)
    y_offset = _semantic.to_tensor(y_offset)
    multicast = ttgl._unwrap_if_constexpr(multicast)
    _semantic.builder.create_async_tma_gather(tensor_desc.handle, x_offsets.handle, y_offset.handle, barrier.handle,
                                              result.handle, pred.handle, multicast)


def _emit_scatter_nonnegative_check(x_offsets, y_offset, _semantic=None):
    y_offset = ttgl.to_tensor(y_offset, _semantic=_semantic)
    zero = ttgl.to_tensor(0, _semantic=_semantic)

    is_nonnegative = y_offset.__ge__(zero, _semantic=_semantic)
    ttgl.device_assert(is_nonnegative, "async_scatter y_offset cannot be negative", _semantic=_semantic)

    is_nonnegative = x_offsets.__ge__(zero, _semantic=_semantic)
    ttgl.device_assert(is_nonnegative, "async_scatter x_offsets cannot have any negative elements", _semantic=_semantic)


@builtin
def async_scatter(tensor_desc, x_offsets, y_offset, src, _semantic=None):
    """
    Asynchronously scatter elements from shared memory to global memory using TMA.

    Args:
        tensor_desc (tensor_descriptor): The tensor descriptor.
        x_offsets (tensor): 1D tensor of X offsets.
        y_offset (int): Scalar Y offset.
        src (tensor_memory_descriptor): The source data, must be in NVMMASharedLayout.
    """
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, (y_offset, ), "async_scatter", "y_offset", _semantic=_semantic)
        _emit_scatter_nonnegative_check(x_offsets, y_offset, _semantic=_semantic)

    y_offset = _semantic.to_tensor(y_offset)
    _semantic.builder.create_async_tma_scatter(tensor_desc.handle, x_offsets.handle, y_offset.handle, src.handle)
<|endoftext|>from ..._semantic import _check
from ..._core import _unwrap_if_constexpr, builtin
from triton._C.libtriton import ir

__all__ = [
    "async_copy_global_to_shared",
    "async_load",
    "mbarrier_arrive",
    "commit_group",
    "wait_group",
]


@builtin
def async_load(smem, pointer, mask=None, cache_modifier="", eviction_policy="", volatile=False, _semantic=None):
    """
    Asynchronously load elements from global memory to shared memory.

    Args:
        smem (shared_memory_descriptor): Destination shared memory descriptor.
        pointer (tensor): Source pointer tensor.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
        eviction_policy (str): Eviction policy specifier. Defaults to "".
        volatile (bool): Whether the load is volatile. Defaults to False.
    """
    mask = _unwrap_if_constexpr(mask)
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction_policy = _semantic._str_to_eviction_policy(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
    _check(
        smem.shape == pointer.shape, lambda:
        f"expected smem shape to match pointer shape but got smem.shape = {smem.shape}, pointer.shape = {pointer.shape}"
    )
    mask_handle = mask.handle if mask is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(smem.handle, pointer.handle, mask_handle, ir.value(),
                                                        cache_modifier, eviction_policy, volatile)


# Backward-compatible alias
async_copy_global_to_shared = async_load


@builtin
def mbarrier_arrive(mbarrier, increment_count=True, _semantic=None):
    """
    Arrive on the mbarrier once all outstanding async copies are complete.

    Args:
        mbarrier (shared_memory_descriptor): Barrier object to arrive on.
        increment_count (bool): Whether to increment the arrival count. Defaults to True.
    """
    increment_count = _unwrap_if_constexpr(increment_count)
    _semantic.builder.create_async_copy_mbarrier_arrive(mbarrier.handle, increment_count)


@builtin
def commit_group(_semantic=None):
    """
    Commit the current asynchronous copy group.

    This finalizes a set of asynchronous copy operations.
    """
    _semantic.builder.create_async_commit_group()


@builtin
def wait_group(num_outstanding=0, _semantic=None):
    """
    Wait for outstanding asynchronous copy group operations.

    Args:
        num_outstanding (int): Wait until `num_outstanding` or less async copy groups in-flight. Defaults to 0.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_wait_group(num_outstanding)
<|endoftext|>from __future__ import annotations

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._layouts import DotOperandLayout, NVMMADistributedLayout
from ..._core import builtin, _unwrap_if_constexpr
from . import async_copy, mbarrier

__all__ = ["async_copy", "mbarrier", "mma_v2"]


@builtin
def mma_v2(a, b, acc, input_precision=None, _semantic=None):
    input_precision = _unwrap_if_constexpr(input_precision)
    assert isinstance(a, ttgl.tensor), "a must be a tensor"
    assert isinstance(b, ttgl.tensor), "b must be a tensor"
    assert isinstance(acc, ttgl.tensor), "acc must be a tensor"

    mma_layout = acc.type.layout
    assert isinstance(mma_layout, NVMMADistributedLayout), "acc must have an NVMMADistributedLayout"
    assert mma_layout.version == [2, 0], "MMA layout must have version 2.0"

    assert isinstance(a.type.layout, DotOperandLayout), "a must have a DotOperandLayout"
    assert isinstance(b.type.layout, DotOperandLayout), "b must have a DotOperandLayout"
    assert a.type.layout.parent == mma_layout, "a's parent layout must be the same as acc's layout"
    assert b.type.layout.parent == mma_layout, "b's parent layout must be the same as acc's layout"
    assert a.type.layout.operand_index == 0, "a's operand index must be 0"
    assert b.type.layout.operand_index == 1, "b's operand index must be 1"

    handle = _semantic.dot(a, b, acc, input_precision=input_precision, max_num_imprecise_acc=None,
                           out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)
<|endoftext|>import math

import triton.experimental.gluon.language as ttgl
from triton.experimental.gluon._runtime import constexpr_function, jit
from triton.experimental.gluon.language._layouts import SwizzledSharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["allocate_mbarrier", "arrive", "init", "invalidate", "MBarrierLayout", "wait"]


class MBarrierLayout(SwizzledSharedLayout):
    """
    Layout for mbarrier synchronization in Ampere and later architectures.

    Args:
        cga_layout (List[List[int]]): CGA layout bases. Defaults to [].
    """

    def __init__(self, cga_layout=None):
        super().__init__(vec=1, per_phase=1, max_phase=1, order=[0], cga_layout=cga_layout or [])

    @staticmethod
    @constexpr_function
    def multicta(num_ctas: int, two_cta: bool = False):
        """
        Create a multi-CTA mbarrier layout.

        Args:
            num_ctas (int): Number of CTAs.
            two_cta (bool): Whether the barrier should synchronize every other CTA
        """
        num_ctas = ttgl._unwrap_if_constexpr(num_ctas)
        two_cta = ttgl._unwrap_if_constexpr(two_cta)
        if two_cta:
            assert num_ctas % 2 == 0, "num_ctas must be even for two-CTA mode"
        assert num_ctas > 0, "num_ctas must be positive"
        assert (num_ctas & (num_ctas - 1)) == 0, "num_ctas must be a power of two"

        bases = []
        if two_cta:
            bases.append([0])
            num_ctas //= 2

        for i in range(int(math.log2(num_ctas))):
            bases.append([2**i])
        return MBarrierLayout(bases)


@jit
def allocate_mbarrier(batch: ttgl.constexpr = None, two_ctas: ttgl.constexpr = False):
    """
    Helper function to allocate an mbarrier

    Args:
        two_ctas (bool): Whether the barrier should synchronize every other CTA
    """
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    num_elems: ttgl.constexpr = num_ctas if not two_ctas else num_ctas // 2
    ttgl.static_assert(batch is None or isinstance(batch.value, int))
    shape: ttgl.constexpr = [num_elems] if batch is None else [batch, num_elems]
    bar = ttgl.allocate_shared_memory(
        ttgl.int64,
        shape,
        MBarrierLayout.multicta(num_ctas=num_ctas, two_cta=two_ctas),
    )
    return bar


@builtin
def init(mbarrier, count, _semantic=None):
    """
    Initialize an mbarrier with a specified count.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to initialize.
        count (int): The initial count for the barrier.
    """
    count = _unwrap_if_constexpr(count)
    _semantic.builder.create_mbarrier_init(mbarrier.handle, count)


@builtin
def invalidate(mbarrier, _semantic=None):
    """
    Invalidate an mbarrier, resetting its state.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to invalidate.
    """
    _semantic.builder.create_mbarrier_inval(mbarrier.handle)


@builtin
def wait(mbarrier, phase, pred=True, deps=(), _semantic=None):
    """
    Wait until the mbarrier object completes its current phase.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to wait on.
        phase (int): The phase index to wait for.
        pred (bool): Predicate. Operation is skipped if predicate is False. Defaults to True.
        deps (Sequence[shared_memory_descriptor]): Dependent allocations barrier is waiting on. Used to track liveness of dependent allocations. Defaults to ().
    """
    phase = _semantic.to_tensor(phase)
    pred = _semantic.to_tensor(pred)
    deps = [x.handle for x in deps]
    _semantic.builder.create_mbarrier_wait(mbarrier.handle, phase.handle, pred.handle, deps)


@builtin
def arrive(mbarrier, *, pred=True, _semantic=None):
    """
    Arrive on an mbarrier, signaling that a thread has reached the barrier.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to arrive on.
        pred (bool): Predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    count = 1
    cta_mask = 0
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, cta_mask, pred.handle)
<|endoftext|>from __future__ import annotations


class warp_pipeline_stage:
    """
    Marks a warp-pipeline stage inside a Gluon kernel.

	When used inside @gl.kernel, each with amd.warp_pipeline_stage(...) block
	semantically defines a distinct stage of a warp pipeline. All operations
	inside the block belong to the same pipeline cluster and are intended to
	execute as a unit relative to other stages.

	The optional string label (e.g., "load", "compute") names the pipeline
	stage for identification and diagnostics, without affecting program
	semantics.

	An optional integer priority may be specified to express the relative
	scheduling priority of the warp the stage belongs to. The priority applies
	to the entire cluster. Valid values range from 0 (lowest) to 3 (highest)
    as it's lowered to the operand of `s_setprio`. If unspecified, priority
    resets to zero when any other stage in the loop uses explicit priority;
    otherwise no priority instruction is emitted.
    N.B., This is a performance hint to the hardware scheduler, and its effect
	may vary depending on the dynamic interaction of instruction streams
	across different warps. It is optional and should be used judiciously,
	only when explicit scheduling guidance is beneficial.

    Example: (only to show how to use, this example is not supposed to
    represent the optimal way.)

    @gl.kernel
    ...

    for k in gl.range(0, K, one):

        # Stage 0: prefetch tiles
        with amd.warp_pipeline_stage("load", priority=3):
            a = gl.amd.buffer_load(a_ptr, offs_a)
            b = gl.amd.buffer_load(b_ptr, offs_b)

        # Stage 1: prepare MFMA operands
        with amd.warp_pipeline_stage("prep"):
            a_tile = a.load(layout=...)
            b_tile = b.load(layout=...)

        # Stage 2: compute
        with amd.warp_pipeline_stage("compute", priority=0):
            acc = gl.amd.mfma(a_tile, b_tile, acc)
            offs_a += strideA
            offs_b += strideB
    """

    __slots__ = ("label", "priority", "_semantic")

    def __init__(self, label=None, *, priority: int | None = None, **_internal):
        self.label = getattr(label, "value", None)
        if priority is not None:
            assert priority > -1 and priority < 4, "priority should be 0 to 3."
        self.priority = priority
        self._semantic = _internal.get("_semantic", None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        if self._semantic is None:
            return False
        marker = self.label if self.label is not None else "cluster"
        prio = self.priority if self.priority is not None else -1
        self._semantic.builder.create_warp_pipeline_border(marker, prio)
        return False
<|endoftext|>from __future__ import annotations

from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check

from .._core import builtin, _unwrap_if_constexpr

__all__ = ["slice"]


@builtin
def slice(source, shape, offsets, _semantic=None):
    """Extract a register only slice of a tensor.

    Returns a view of ``source`` of the requested ``shape`` starting at
    ``offsets``, keeping the source's distributed layout. Because the layout is
    preserved, the slice introduces no cross-thread data movement; the slice
    extent and offsets must align to the source layout's CTA tiling.

    Args:
        source (tensor): The tensor to slice. Must have a distributed layout.
        shape (List[int]): Shape of the extracted slice (same rank as source).
        offsets (List[int]): Per-dimension start offsets into ``source``.
    """
    _check(isinstance(source.type, ttgl.distributed_type),
           lambda: f"Expected source to have a distributed_type but got {source.type}")

    shape = [_unwrap_if_constexpr(s) for s in shape]
    offsets = [_unwrap_if_constexpr(o) for o in offsets]

    rank = len(source.type.shape)
    _check(len(shape) == rank, lambda: f"slice: shape must have rank {rank}, got {len(shape)}")
    _check(len(offsets) == rank, lambda: f"slice: offsets must have rank {rank}, got {len(offsets)}")

    src_shape = source.type.shape
    for i in range(rank):
        _check(shape[i] <= src_shape[i], lambda: f"slice: result shape {shape} cannot exceed source shape {src_shape}")
        _check(offsets[i] + shape[i] <= src_shape[i],
               lambda: f"slice: offset {offsets} + shape {shape} exceeds source shape {src_shape}")

    ret_ty = ttgl.distributed_type(source.dtype, shape, source.type.layout)
    handle = _semantic.builder.create_extract_slice(ret_ty.to_ir(_semantic.builder), source.handle, offsets)
    return ttgl.tensor(handle, ret_ty)
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from triton.language.core import _unwrap_if_constexpr

from triton.experimental.gluon.language._layouts import DistributedLayout

__all__ = [
    "AMDMFMALayout",
    "AMDWMMALayout",
]


@dataclass(frozen=True)
class AMDMFMALayout(DistributedLayout):
    """
    Represents a layout for AMD MFMA (matrix core) operations.

    Args:
        version (int): The GPU architecture.
        instr_shape (List[int]): The shape in the form of (M, N, K) of the matrix.
        transposed (bool): Indicates the result tensor is transposed so that each thread holds consecutive elements in the same row instead of column, which is good for chained dot and global write.
        warps_per_cta (List[int]): The warp layout in the block.
        element_bitwidth Optional(int): Bit width of the output element type. Supported values are 32 and 64. Defaults to 32.
        tiles_per_warp Optional(List[int]): The tile layout within a warp. Defaults to unit tile layout, i.e., single tile on all dimensions.
        cga_layout (Optional[List[List[int]]]): Bases describing CTA tiling.

    Current supported versions:

    - 1: gfx908
    - 2: gfx90a
    - 3: gfx942
    - 4: gfx950
    """
    version: int
    instr_shape: List[int]
    transposed: bool
    warps_per_cta: List[int]
    element_bitwidth: Optional[int] = None
    tiles_per_warp: Optional[List[int]] = None
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("version", _unwrap_if_constexpr(self.version))
        super().__setattr__("instr_shape", _unwrap_if_constexpr(self.instr_shape))
        super().__setattr__("transposed", _unwrap_if_constexpr(self.transposed))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("element_bitwidth", _unwrap_if_constexpr(self.element_bitwidth))
        super().__setattr__("tiles_per_warp", _unwrap_if_constexpr(self.tiles_per_warp))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))

        if self.element_bitwidth is None:
            super().__setattr__("element_bitwidth", 32)
        if self.tiles_per_warp is None:
            super().__setattr__("tiles_per_warp", [1] * len(self.warps_per_cta))

        self.verify()

    def _to_ir(self, builder):
        return builder.get_amd_mfma_layout(
            self.version,
            self.warps_per_cta,
            self.instr_shape,
            self.transposed,
            self.cga_layout,
            self.tiles_per_warp,
            self.element_bitwidth,
        )

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        cga_layout = stringify(["~".join(map(str, vec)) for vec in self.cga_layout] if self.cga_layout else None)
        return f"MFMA_{self.version}_{stringify(self.instr_shape)}_{self.transposed}_{stringify(self.warps_per_cta)}_{self.element_bitwidth}_{stringify(self.tiles_per_warp)}_{cga_layout}_MFMA"

    def verify(self):
        assert self.version >= 1 and self.version <= 4, "version must be in the [1, 4] range"
        assert len(self.instr_shape) == 3, "instr_shape must follow the (M, N, K) format"
        valid_shapes = [[32, 32], [16, 16], [64, 4], [4, 64]]
        assert self.instr_shape[0:2] in valid_shapes, f"invalid intrinsic shape {self.instr_shape}"
        assert self.element_bitwidth in [32, 64], "element bitwidth must be 32 or 64"

        rank = len(self.warps_per_cta)
        assert all(len(vec) == rank for vec in self.cga_layout), "cga_layout basis rank mismatch"

    def __hash__(self):
        return hash((
            self.version,
            tuple(self.instr_shape),
            self.transposed,
            tuple(self.warps_per_cta),
            self.element_bitwidth if self.element_bitwidth else None,
            tuple(self.tiles_per_warp) if self.tiles_per_warp else None,
            tuple(tuple(vec) for vec in self.cga_layout),
        ))

    @property
    def rank(self):
        return len(self.warps_per_cta)


@dataclass(frozen=True)
class AMDWMMALayout(DistributedLayout):
    """
    Represents a layout for AMD WMMA (matrix core) operations.

    Args:
        version (int): Indicates the GPU architecture.
        transposed (bool): Indicates the result tensor is transposed.
        warp_bases (List[List[int]]): Warp bases for CTA layout.
        reg_bases (Optional[List[List[int]]]): Repetition (register) bases for CTA layout.
        instr_shape (Optional[List[int]]): Instruction shape (M, N, K). Defaults to (16, 16, 16).
        cga_layout (Optional[List[List[int]]]): Bases describing CTA tiling.
        rank (Optional[int]): rank of warp and register bases. Default to 2 if missing.

    Current supported versions:

    - 1: RDNA3; e.g., gfx1100, gfx1101
    - 2: RDNA4; e.g., gfx1200, gfx1201
    - 3: gfx1250
    """
    version: int
    transposed: bool
    warp_bases: List[List[int]]
    reg_bases: Optional[List[List[int]]] = None
    instr_shape: Optional[List[int]] = None
    cga_layout: List[List[int]] = field(default_factory=list)
    rank: Optional[int] = None

    def __post_init__(self):
        super().__setattr__("version", _unwrap_if_constexpr(self.version))
        super().__setattr__("transposed", _unwrap_if_constexpr(self.transposed))
        super().__setattr__("warp_bases", [list(inner) for inner in _unwrap_if_constexpr(self.warp_bases)])
        super().__setattr__("reg_bases",
                            [list(inner)
                             for inner in _unwrap_if_constexpr(self.reg_bases)] if self.reg_bases is not None else [])
        instr_shape = _unwrap_if_constexpr(self.instr_shape) if self.instr_shape is not None else [16, 16, 16]
        super().__setattr__("instr_shape", _unwrap_if_constexpr(instr_shape))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))
        rank = _unwrap_if_constexpr(self.rank) if self.rank is not None else 2
        super().__setattr__("rank", rank)
        self.verify()

    def _to_ir(self, builder):
        return builder.get_amd_wmma_layout(
            self.version,
            self.transposed,
            self.warp_bases,
            self.reg_bases,
            self.cga_layout,
            self.instr_shape,
            self.rank,
        )

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        def nested_stringify(x):
            return stringify(["~".join(map(str, vec)) for vec in x] if x else None)

        warp_bases = nested_stringify(self.warp_bases)
        reg_bases = nested_stringify(self.reg_bases)
        cga_layout = nested_stringify(self.cga_layout)
        return f"WMMA_{self.version}_{self.transposed}_{warp_bases}_{reg_bases}_{stringify(self.instr_shape)}_{cga_layout}_{self.rank}_WMMA"

    def verify(self):
        assert self.version >= 1 and self.version <= 3, "version must be in the [1, 3] range"
        if len(self.warp_bases) > 0:
            assert len(self.warp_bases[0]) == self.rank, "warp_bases basis rank mismatch"
        assert all(len(vec) == self.rank for vec in self.cga_layout), "cga_layout basis rank mismatch"

    def __hash__(self):
        return hash((
            self.version,
            self.transposed,
            tuple(tuple(vec) for vec in self.warp_bases),
            tuple(tuple(vec) for vec in self.reg_bases),
            tuple(self.instr_shape) if self.instr_shape else None,
            tuple(tuple(vec) for vec in self.cga_layout),
            self.rank,
        ))
<|endoftext|>from .._core import builtin
from ._layouts import AMDMFMALayout, AMDWMMALayout
from . import cdna3, cdna4
from . import rdna3, rdna4
from . import gfx1250
from . import cdna5
from .slice import slice
from .warp_pipeline import warp_pipeline_stage

__all__ = [
    "AMDMFMALayout", "AMDWMMALayout", "cdna3", "cdna4", "cdna5", "rdna3", "rdna4", "gfx1250", "warp_pipeline_stage",
    "slice"
]
<|endoftext|>import math

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check

from .._core import _unwrap_if_constexpr
from .._layouts import DotOperandLayout
from ._layouts import AMDWMMALayout


def _wrap_scaled_upcast_result(handle, elem_type, semantic):
    shape = semantic.builder.get_shape_from_tensor(handle)
    layout = semantic.builder.get_gluon_layout_from_tensor(handle)
    ret_ty = ttgl.distributed_type(elem_type, shape, layout)
    return ttgl.tensor(handle, ret_ty)


def _verify_wmma(version, a, b, acc):
    _check(acc is not None, lambda: "acc is required")

    layout = acc.type.layout
    _check(
        isinstance(layout, AMDWMMALayout) and layout.version == version,
        lambda: f"Expected layout to be an instance of AMDWMMALayout with version {version}")

    a_layout = a.type.layout
    _check(
        isinstance(a_layout, DotOperandLayout) and isinstance(a_layout.parent, AMDWMMALayout)
        and a_layout.parent.version == version,
        lambda: "Expected a's layout to be a DotOperandLayout with parent matching AMDWMMALayout")

    b_layout = b.type.layout
    _check(
        isinstance(b_layout, DotOperandLayout) and isinstance(b_layout.parent, AMDWMMALayout)
        and b_layout.parent.version == version,
        lambda: "Expected b's layout to be a DotOperandLayout with parent matching AMDWMMALayout")


def _wmma(version, a, b, acc, semantic):
    """ Shared implementation for AMD WMMA operations for Gluon builtins """
    _verify_wmma(version, a, b, acc)

    handle = semantic.dot(a, b, acc, input_precision=knobs.language.fp32_default, max_num_imprecise_acc=None,
                          out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)


def _mma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, scale_fn, semantic):
    """ Shared implementation for AMD WMMA scaled and MFMA scaled operation. """

    def _get_scale_shape(op_idx, operand, format, scale_factor):
        operand_shape = [s for s in operand.type.shape]
        scale_shape = operand_shape
        unpack_factor = 2 if format == "e2m1" else 1
        if op_idx == 0:
            k = scale_shape[-1] * unpack_factor
            scale_shape[-1] = k // scale_factor
        else:
            k = scale_shape[-2] * unpack_factor
            scale_shape[-2] = k // scale_factor
            scale_shape[-2], scale_shape[-1] = scale_shape[-1], scale_shape[-2]
        return scale_shape

    def _get_default_scale_dtype_and_unit_value(op_idx):
        default_value_by_dtype = {ttgl.uint8: 0x7F, ttgl.float8e4nv: 1.0}

        if a_scale is None and b_scale is None:
            return ttgl.uint8, 0x7F

        if a_format == b_format == "e2m1":
            # Fp4 x Fp4 requries to use the same scale dtype for both operands.
            other_scale = b_scale if op_idx == 0 else a_scale
            return other_scale.dtype, default_value_by_dtype[other_scale.dtype]

        return ttgl.uint8, 0x7F

    def _create_and_broadcast_default_scale(op_idx, scale, format, scale_factor):
        operand = a if op_idx == 0 else b

        scale_shape = _get_scale_shape(op_idx, operand, format, scale_factor)
        if isinstance(scale, ttgl.tensor) and scale.numel.value != 1:
            # In the case of scale pre-shuffling, the input shape is different from the default shape. We only check
            # the number of elements here.
            assert math.prod(scale_shape) == scale.numel.value, "Incompatible scale shape"
            return scale

        scale_layout = scale_fn(operand.type.layout, scale_shape, scale_factor)
        scale_value = _unwrap_if_constexpr(scale)
        if scale_value is None:
            scale_dtype, scale_value = _get_default_scale_dtype_and_unit_value(op_idx)
        elif isinstance(scale_value, int):
            scale_dtype = ttgl.uint8
        elif isinstance(scale_value, float):
            scale_dtype = ttgl.float8e4nv
        else:
            scale_dtype = scale.dtype

        return semantic.full(scale_shape, scale_value, scale_dtype, scale_layout)

    scale_factor = semantic.deduce_scale_factor(a, a_scale, a_format, True, b, b_scale, b_format, True)

    a_scale = _create_and_broadcast_default_scale(0, a_scale, a_format, scale_factor)
    b_scale = _create_and_broadcast_default_scale(1, b_scale, b_format, scale_factor)
