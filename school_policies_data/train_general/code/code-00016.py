        return tl_obj_load_amd(obj.desc, offsets)
    elif isinstance(obj, amd_tdm.tensor_descriptor):
        return tl_obj_load_amd(obj, offsets)
    else:
        return obj.load(offsets)


@gluon.jit
def tl_obj_gather_amd(desc_args, x_offsets, y_offset):
    NUM_IDX: ttgl.constexpr = x_offsets.shape[0]
    BLOCK_N: ttgl.constexpr = desc_args.desc.block_shape[1]
    smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    gather_block_shape: ttgl.constexpr = [NUM_IDX, BLOCK_N]
    gather_desc = amd_tdm.make_tensor_descriptor(desc_args.base_ptr, desc_args.desc.shape, desc_args.desc.strides,
                                                 gather_block_shape, smem_layout)
    num_warps: ttgl.constexpr = ttgl.num_warps()
    gather_shape: ttgl.constexpr = gather_desc.block_shape
    idx_base: ttgl.constexpr = ttgl.BlockedLayout([gather_shape[0], 1],
                                                  [1, get_num_threads_per_warp(current_target())], [1, num_warps],
                                                  [1, 0])
    idx_layout: ttgl.constexpr = ttgl.SliceLayout(1, idx_base)
    x_offsets = ttgl.convert_layout(x_offsets, idx_layout)
    alloc = ttgl.allocate_shared_memory(desc_args.desc.dtype, list(gather_shape), smem_layout)
    y_off = ttgl.to_tensor(y_offset)
    gather_desc = amd_tdm.update_tensor_descriptor(gather_desc, add_offsets=[0, y_off], clamp_bounds=True)
    amd_tdm.async_gather(gather_desc, x_offsets, alloc)
    amd_tdm.async_wait(0)
    ret_layout: ttgl.constexpr = default_blocked_layout(list(gather_shape), num_warps, current_target())
    out = alloc.load(ret_layout)
    return out


@gluon.jit
def tl_obj_scatter_amd(desc_args, value, x_offsets, y_offset):
    NUM_IDX: ttgl.constexpr = x_offsets.shape[0]
    BLOCK_N: ttgl.constexpr = desc_args.desc.block_shape[1]
    smem_layout: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    scatter_block_shape: ttgl.constexpr = [NUM_IDX, BLOCK_N]
    scatter_desc = amd_tdm.make_tensor_descriptor(desc_args.base_ptr, desc_args.desc.shape, desc_args.desc.strides,
                                                  scatter_block_shape, smem_layout)
    num_warps: ttgl.constexpr = ttgl.num_warps()
    scatter_shape: ttgl.constexpr = scatter_desc.block_shape
    idx_base: ttgl.constexpr = ttgl.BlockedLayout([scatter_shape[0], 1],
                                                  [1, get_num_threads_per_warp(current_target())], [1, num_warps],
                                                  [1, 0])
    idx_layout: ttgl.constexpr = ttgl.SliceLayout(1, idx_base)
    x_offsets = ttgl.convert_layout(x_offsets, idx_layout)
    alloc = ttgl.allocate_shared_memory(desc_args.desc.dtype, list(scatter_shape), smem_layout, value)
    y_off = ttgl.to_tensor(y_offset)
    scatter_desc = amd_tdm.update_tensor_descriptor(scatter_desc, add_offsets=[0, y_off], clamp_bounds=True)
    amd_tdm.async_scatter(scatter_desc, x_offsets, alloc)
    amd_tdm.async_wait(0)


@gluon.jit
def tl_obj_gather(obj, x_offsets, y_offset):
    if isinstance(obj, AMDTensorDescriptorArgs):
        return tl_obj_gather_amd(obj, x_offsets, y_offset)
    else:
        return obj.gather(x_offsets, y_offset)


@gluon.jit
def tl_obj_scatter(obj, value, x_offsets, y_offset):
    if isinstance(obj, AMDTensorDescriptorArgs):
        tl_obj_scatter_amd(obj, value, x_offsets, y_offset)
    else:
        obj.scatter(value, x_offsets, y_offset)


# ---- AMD host-side descriptor ----


def convert_host_descriptor(desc):
    from triton.tools.tensor_descriptor import TensorDescriptor

    assert isinstance(desc, TensorDescriptor)
    block_shape = desc.block_shape
    tensor = desc.base

    layout = get_default_tdm_layout(*block_shape)
    return gluon.amd.gfx1250.TensorDescriptor(tensor, list(desc.shape), list(desc.strides), block_shape, layout)
<|endoftext|>import argparse
import ast
import builtins
import inspect
from dataclasses import dataclass, field
from types import FunctionType, ModuleType
from typing import Any, cast

import triton.language as tl  # type: ignore[import-untyped]
from triton.runtime.jit import JITCallable, JITFunction  # type: ignore[import-untyped]
from triton.tools.triton_to_gluon_translator.ordered_set import ordered_set
from triton.tools.triton_to_gluon_translator.scoped_dict import scoped_dict
from triton.tools.triton_to_gluon_translator.slice_kernel import (
    GlobalValue,
    ReferenceRewriter,
    RewriteFn,
    add_sugar_rewrites,
    find_references,
    get_base_value,
    is_submodule,
    mangle_reference_names,
    parse_expr,
)
from triton.tools.triton_to_gluon_translator.target import TranslatorTarget
from triton.tools.triton_to_gluon_translator.stable_toposort import stable_toposort


def one_to_one_rewrite(obj: Any) -> RewriteFn:

    def rewrite(global_value: GlobalValue, imports: ordered_set[str]) -> ast.AST | None:
        if global_value.original_value is obj:
            return ast.Attribute(value=ast.Name(id="gl", ctx=ast.Load()), attr=obj.__name__, ctx=ast.Load())
        return None

    return rewrite


def add_one_to_one_rewrites(rewrites: list[RewriteFn]) -> None:
    import triton.experimental.gluon.language as gl  # type: ignore[import-untyped]

    for value in vars(gl).values():
        module = inspect.getmodule(value)
        if module is None:
            continue
        if getattr(value, "__triton_builtin__", False) and is_submodule(module, ["triton.language"]):
            tl_value = getattr(tl, value.__name__, None)
            if tl_value is None:
                tl_value = getattr(tl.core, value.__name__, None)
            if tl_value is None:
                continue
            rewrites.append(one_to_one_rewrite(tl_value))
        elif isinstance(value, JITFunction):
            tl_value = getattr(tl, value.fn.__name__, None)
            if tl_value is None:
                tl_value = getattr(tl.standard, value.fn.__name__, None)
            if tl_value is None:
                continue
            assert isinstance(tl_value, JITFunction) and value is not tl_value
            if value.fn is tl_value.fn:
                rewrites.append(one_to_one_rewrite(tl_value))


def translator_helper_rewrite(obj: Any, helper_name: str) -> RewriteFn:

    def rewrite(global_value: GlobalValue, imports: ordered_set[str]) -> ast.AST | None:
        if global_value.original_value is obj:
            return ast.Attribute(value=ast.Name(id="helpers", ctx=ast.Load()), attr=helper_name, ctx=ast.Load())
        return None

    return rewrite


def add_translator_helper_rewrites(rewrites: list[RewriteFn]) -> None:
    remap: list[tuple[Any, str]] = [
        (tl.arange, "tl_arange"),
        (tl.full, "tl_full"),
        (tl.trans, "tl_trans"),
        (tl.cat, "tl_cat"),
        (tl.dot, "tl_dot"),
        (tl.dot_scaled, "tl_dot_scaled"),
        (tl.make_tensor_descriptor, "tl_make_tensor_descriptor"),
        (tl.load_tensor_descriptor, "tl_load_tensor_descriptor"),
        (tl.store_tensor_descriptor, "tl_store_tensor_descriptor"),
    ]
    if (tl_cuda := getattr(tl.extra, "cuda", None)) is not None:
        remap.append((tl_cuda.num_threads, "get_num_threads_per_program"))
    for value, helper_name in remap:
        rewrites.append(translator_helper_rewrite(value, helper_name))


def expr_rewrite(obj: Any, expr: str) -> RewriteFn:

    def rewrite(value: GlobalValue, imports: ordered_set[str]) -> ast.AST | None:
        if value.original_value is obj:
            return parse_expr(expr)
        return None

    return rewrite


def add_expr_rewrites(rewrites: list[RewriteFn]) -> None:
    import triton
    import triton.language as tl

    rewrites.append(expr_rewrite(triton.jit, "gluon.jit"))
    rewrites.append(expr_rewrite(tl.debug_barrier, "gl.barrier"))


@dataclass
class Translator(ReferenceRewriter):
    target: TranslatorTarget = field(kw_only=True)
    tensor_member_match_fns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        import triton
        import triton.language as tl

        self.context.setdefault("tl", tl)
        self.context.setdefault("triton", triton)

        add_sugar_rewrites(self.rewrites, translate_to_gluon=True)
        add_translator_helper_rewrites(self.rewrites)
        add_one_to_one_rewrites(self.rewrites)
        add_expr_rewrites(self.rewrites)

        self.imports.add("import triton.experimental.gluon as gluon")
        self.imports.add("import triton.experimental.gluon.language as gl")
        self.imports.add(f"import {self.target.helpers_module} as helpers")

        self.tensor_member_match_fns = ["reshape", "trans", "permute", "split", "reduce", "sum", "expand_dims"]

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        new_node = super().visit_Attribute(node)
        if node.attr == "T":
            new_node = parse_expr(f"helpers.reset_to_default_layout({ast.unparse(new_node)})")
        return new_node

    def canonicalize_call(self, node: ast.Call) -> tuple[ast.Call, str | None]:
        if self.get_reference(node.func) is not None:
            return node, None
        if (not isinstance(node.func, ast.Attribute) or node.func.attr not in self.tensor_member_match_fns):
            return node, None
        new_callable = parse_expr(f"tl.{node.func.attr}")
        new_call = ast.Call(func=new_callable, args=[node.func.value] + node.args, keywords=node.keywords)
        return new_call, node.func.attr

    def uncanonicalize_call(self, node: ast.Call, fn_name: str | None) -> ast.Call:
        if fn_name is None:
            return node
        value = node.args[0]
        new_callable = ast.Attribute(value, fn_name, ctx=ast.Load())
        return ast.Call(func=new_callable, args=node.args[1:], keywords=node.keywords)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node, canonicalized = self.canonicalize_call(node)
        ref = self.get_reference(node.func)
        if ref is None:
            assert canonicalized is None
            if isinstance(node.func, ast.Attribute) and node.func.attr in [
                    "store",
                    "load",
                    "gather",
                    "scatter",
            ]:
                new_callee = parse_expr(f"helpers.tl_obj_{node.func.attr}")
                node = ast.Call(func=new_callee, args=[node.func.value] + node.args, keywords=node.keywords)
            return self.generic_visit(node)
        value, _, _ = ref
        if value in [tl.reshape, tl.ravel]:
            node.keywords = [kw for kw in node.keywords if kw.arg != "can_reorder"]
        elif value is tl.split:
            node.args[0] = parse_expr(f"helpers.set_split_src_layout({ast.unparse(node.args[0])})")
        elif value is tl.expand_dims:
            node.args[0] = parse_expr(
                f"helpers.convert_to_expand_dims_layout({ast.unparse(node.args[0])}, [{ast.unparse(node.args[1])}])")
        elif value is tl.range:
            return ast.Call(
                func=ast.Name("range", ast.Load()),
                args=[cast(ast.expr, self.generic_visit(arg)) for arg in node.args],
                keywords=[],
            )

        node = self.uncanonicalize_call(node, canonicalized)
        new_node = self.generic_visit(node)
        if value in [tl.reshape, tl.trans, tl.permute, tl.join, tl.split, tl.reduce, tl.sum, tl.max, tl.min]:
            new_node = cast(ast.Call, parse_expr(f"helpers.reset_to_default_layout({ast.unparse(new_node)})"))
        return new_node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if not isinstance(node.slice, ast.Tuple):
            return self.generic_visit(node)

        expand_dims: list[int] = []
        for index, dim in enumerate(node.slice.elts):
            if isinstance(dim, ast.Constant) and dim.value is None:
                expand_dims.append(index)
            elif isinstance(dim, ast.Slice) and all(d is None for d in [dim.lower, dim.upper, dim.step]):
                continue
            else:
                return self.generic_visit(node)
        value_expr = parse_expr(f"helpers.convert_to_expand_dims_layout({ast.unparse(node.value)}, {expand_dims})")
        node = ast.Subscript(value=value_expr, slice=node.slice, ctx=node.ctx)
        return self.generic_visit(node)


def translate_kernels(kernels: list[GlobalValue], target: TranslatorTarget) -> str:

    def filter(value: ModuleType | GlobalValue) -> bool:
        if isinstance(value, ModuleType):
            return False
        if getattr(value.original_value, "__triton_builtin__", False):
            return True
        if isinstance(value.original_value, JITFunction):
            if getattr(tl.tensor, value.name, None) is value.original_value:
                return True
            if value.original_value.is_gluon():
                return True
            return False
        assert isinstance(value.original_value, object)
        if isinstance(value.original_value, type | FunctionType | JITCallable):
            return True
        if isinstance(value.original_value, int | float | tl.constexpr):
            return False
        return True

    references, graph = find_references(kernels, filter, value_remap={})
    mangle_reference_names(references, filter)

    ordered_ids = stable_toposort(graph)

    output = ""
    imports: ordered_set[str] = ordered_set()

    ordered_ids = stable_toposort(graph)
    for ref_id in reversed(ordered_ids):
        reference = references[ref_id]
        if filter(reference.value.module) or filter(reference.value):
            continue
        tree = reference.value.parse_ast()
        context = reference.value.get_contextual_defs()
        rewriter = Translator(
            reference.value.module,
            scoped_dict(context),
            references,
            imports,
            filter,
            value_remap={},
            target=target,
        )
        tree = rewriter.visit(tree)
        source = ast.unparse(tree)
        assert reference.mangled_name is not None
        source = reference.value.mangle_source(source, reference.mangled_name)
        output += source + "\n\n\n"
    output = "\n".join(imports) + "\n\n" + output
    return output


def translate_paths(kernel_paths: list[str], target: TranslatorTarget) -> str:
    kernels = [get_base_value(kernel_path) for kernel_path in kernel_paths]
    return translate_kernels(kernels, target=target)


def convert_triton_to_gluon(src: list[JITCallable], target: TranslatorTarget) -> str:

    def wrap_global(kernel):
        name = getattr(getattr(kernel, "fn", kernel), "__name__", "")
        return GlobalValue.wrap(kernel, name, lambda: (name, builtins))

    kernels = [wrap_global(kernel) for kernel in src]
    return translate_kernels(kernels, target=target)


def main(kernels: list[str], output_path: str, target: TranslatorTarget) -> None:
    output = translate_paths(kernels, target=target)
    with open(output_path, "w") as f:
        f.write(output)


def _main_cli() -> None:
    parser = argparse.ArgumentParser(description="Translate Triton kernels to Gluon source.")
    parser.add_argument("kernels", nargs="+", help="Kernel symbols in module.path:object format.")
    parser.add_argument("--output-path", required=True, help="Path to write the translated source.")
    parser.add_argument("--target", required=True, help="Target architecture (e.g. nvidia, gfx1250).")
    args = parser.parse_args()
    main(args.kernels, args.output_path, target=TranslatorTarget(args.target))


if __name__ == "__main__":
    _main_cli()
<|endoftext|>_torch_dtype_to_triton_def = R"""
def _torch_dtype_to_triton(dtype):
    import torch

    if dtype == torch.float8_e5m2:
        return gl.float8e5
    if dtype == torch.float8_e4m3fn:
        return gl.float8e4nv
    return getattr(gl, str(dtype).split(".")[1])
"""

defs: dict[str, str] = {
    "convert_host_descriptor":
    _torch_dtype_to_triton_def + R"""
def convert_host_descriptor(desc):
    from triton.tools.tensor_descriptor import TensorDescriptor

    assert isinstance(desc, TensorDescriptor)
    block_shape = desc.block_shape
    dtype = desc.base.dtype
    tensor = desc.base
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, _torch_dtype_to_triton(dtype))
    return gluon.nvidia.hopper.TensorDescriptor(
        tensor, desc.shape, desc.strides, block_shape, layout
    )
""",
    "convert_host_descriptor_amd":
    _torch_dtype_to_triton_def + R"""
def convert_host_descriptor(desc):
    from triton.tools.tensor_descriptor import TensorDescriptor

    assert isinstance(desc, TensorDescriptor)
    block_shape = desc.block_shape
    dtype = desc.base.dtype
    layout = gl.PaddedSharedLayout.with_identity_for(
        [[block_shape[-1], 4]], list(block_shape), [1, 0]
    )
    return gluon.amd.gfx1250.TensorDescriptor(
        desc.base, list(desc.shape), list(desc.strides), block_shape, layout
    )
""",
}
<|endoftext|>from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class scoped_dict(Generic[K, V]):
    stack: list[dict[K, V]] = field(default_factory=list)

    def __init__(self, d: dict[K, V] | None = None) -> None:
        self.stack = [d or {}]

    def __getitem__(self, key: K) -> V:
        for d in reversed(self.stack):
            if key in d:
                return d[key]
        raise KeyError(key)

    def __setitem__(self, key: K, value: V) -> None:
        self.stack[-1][key] = value

    def __contains__(self, key: K) -> bool:
        return any(key in d for d in reversed(self.stack))

    def setdefault(self, key: K, value: V) -> V:
        return self.stack[-1].setdefault(key, value)

    @contextmanager
    def scope(self, d: dict[K, V] | None = None) -> Generator[None, None, None]:
        self.stack.append(d or {})
        try:
            yield
        finally:
            self.stack.pop()
<|endoftext|>from __future__ import annotations

from enum import Enum


class TranslatorTarget(str, Enum):
    """Target architecture for the Triton-to-Gluon translator.

    Known targets are listed as explicit members for discoverability.
    Unknown ``gfx*`` strings are accepted via ``_missing_()`` so that
    new AMD architectures work without adding an enum member.
    """

    GENERIC = "generic"
    SM80 = "sm80"
    SM90 = "sm90"
    SM100 = "sm100"
    SM103 = "sm103"
    SM107 = "sm107"
    # AMD targets currently exercised by the translator test suite:
    GFX90A = "gfx90a"
    GFX1250 = "gfx1250"
    GFX942 = "gfx942"
    GFX950 = "gfx950"

    @classmethod
    def _missing_(cls, value: object) -> "TranslatorTarget | None":
        if value not in cls._value2member_map_:
            return None
        if isinstance(value, str):
            return cls(value)
        return None

    @property
    def is_amd(self) -> bool:
        return self in (
            TranslatorTarget.GFX90A,
            TranslatorTarget.GFX942,
            TranslatorTarget.GFX950,
            TranslatorTarget.GFX1250,
        )

    @property
    def is_nvidia(self) -> bool:
        return self in (
            TranslatorTarget.SM80,
            TranslatorTarget.SM90,
            TranslatorTarget.SM100,
            TranslatorTarget.SM103,
            TranslatorTarget.SM107,
        )

    @property
    def tensor_descriptor_import(self) -> str:
        module = "amd.gfx1250.tdm" if self.is_amd else "nvidia.hopper.tma"
        return f"from triton.experimental.gluon.language.{module} import tensor_descriptor"

    @property
    def helpers_module(self) -> str:
        base = "triton.tools.triton_to_gluon_translator"

        if self.is_amd:
            return f"{base}.amd_helpers"

        if self.is_nvidia:
            if self in (TranslatorTarget.SM100, TranslatorTarget.SM103, TranslatorTarget.SM107):
                return f"{base}.blackwell_helpers"
            if self in (TranslatorTarget.SM90):
                return f"{base}.hopper_helpers"
            if self in (TranslatorTarget.SM80):
                return f"{base}.nvidia_helpers"

        return f"{base}.common_helpers"
<|endoftext|>from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import logging
import re
import sys
import sysconfig
import tempfile
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any, Callable, TypeAlias

import triton  # type: ignore[import-untyped]
from triton import language as tl  # type: ignore[import-untyped]
from triton.runtime.jit import JITCallable, JITFunction  # type: ignore[import-untyped]
from triton.tools.ragged_tma import create_ragged_descriptor  # type: ignore[import-untyped]
from triton.tools.tensor_descriptor import TensorDescriptor  # type: ignore[import-untyped]
from triton.tools.triton_to_gluon_translator.inline_helpers import defs as inline_helper_defs
from triton.tools.triton_to_gluon_translator.ordered_set import ordered_set
from triton.tools.triton_to_gluon_translator.scoped_dict import scoped_dict
from triton.tools.triton_to_gluon_translator.stable_toposort import stable_toposort
from triton.tools.triton_to_gluon_translator.target import TranslatorTarget

logger = logging.getLogger(__name__)


@dataclass
class GlobalVariable:
    name: str
    value: Any
    module: ModuleType


@dataclass
class GlobalValue:
    value: GlobalVariable | BuiltinFunctionType | FunctionType | type
    original_value: Any

    @staticmethod
    def wrap(
        value: Any,
        name: str,
        resolve_definition: Callable[[], tuple[str, ModuleType]],
    ) -> "GlobalValue":
        assert not isinstance(value, GlobalValue), "value is already a GlobalValue"
        if isinstance(value, FunctionType) and hasattr(value, "__triton_builtin__"):
            return GlobalValue(value, value)
        if isinstance(value, FunctionType) and hasattr(value, "cls"):
            return GlobalValue(value, value)
        # Treat closure globals as global variables, not function definitions.
        if isinstance(value, FunctionType) and value.__closure__ is not None:
            name, module = resolve_definition()
            return GlobalValue(GlobalVariable(name, value, module), value)

        if isinstance(value, BuiltinFunctionType | FunctionType | type):
            return GlobalValue(value, value)
        if isinstance(value, JITCallable):
            assert isinstance(value.fn, FunctionType)
            return GlobalValue(value.fn, value)
        name, module = resolve_definition()
        return GlobalValue(GlobalVariable(name, value, module), value)

    @property
    def name(self) -> str:
        if isinstance(self.value, BuiltinFunctionType | FunctionType | type):
            return self.value.__name__
        assert isinstance(self.value, GlobalVariable)
        return self.value.name

    @property
    def module(self) -> ModuleType:
        if isinstance(self.value, BuiltinFunctionType | FunctionType | type):
            module = inspect.getmodule(self.value)
            assert module is not None, "value is missing module"
            return module
        assert isinstance(self.value, GlobalVariable)
        return self.value.module

    @property
    def id(self) -> int:
        return id(self.original_value)

    def get_contextual_defs(self) -> dict[str, Any]:
        assert not isinstance(self.value, BuiltinFunctionType), "builtin function cannot be scanned"
        if isinstance(self.value, FunctionType):
            # If the function is wrapped, retrieve the original globals to avoid polluting the namespace.
            value = self.value
            while (wrapped := getattr(value, "__wrapped__", None)) is not None:
                value = wrapped
            return value.__globals__
        return self.module.__dict__

    def parse_ast(self) -> ast.AST:
        assert not isinstance(self.value, BuiltinFunctionType), "builtin function cannot be parsed"
        if isinstance(self.value, type | FunctionType):
            return ast.parse(inspect.getsource(self.value))

        assert isinstance(self.value, GlobalVariable), "expected global variable"
        source = inspect.getsource(self.module)
        tree = ast.parse(source)
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign | ast.AnnAssign):
                continue
            target = get_assign_target(stmt)
            if target is not None and target.id == self.value.name:
                assert stmt.value is not None, "FIXME: global variable value is missing"
                return stmt.value
        raise ValueError(f"could not find definition of {self.value} in {self.module}")

    def mangle_source(self, source: str, mangled_name: str) -> str:
        # HACK: An AST rewrite would be more robust, but this works for now.
        if isinstance(self.value, FunctionType):
            return source.replace(f"def {self.name}(", f"def {mangled_name}(")
        elif isinstance(self.value, type):
            return source.replace(f"class {self.name}", f"class {mangled_name}")
        else:
            assert isinstance(self.value, GlobalVariable), "expected global variable"
            return f"{mangled_name} = {source}"


FilterFn = Callable[[ModuleType | GlobalValue], bool]
DecoratorMatcher: TypeAlias = Callable[[scoped_dict[str, Any], ModuleType, ast.expr], bool]
AnnotationRewriter: TypeAlias = Callable[[scoped_dict[str, Any], ModuleType, ast.Subscript], ast.expr | None]


@dataclass
class RewriteSpec:
    ignored_decorator_matchers: Sequence[DecoratorMatcher] = field(default_factory=tuple)
    annotation_rewriters: Sequence[AnnotationRewriter] = field(default_factory=tuple)


def get_assign_target(stmt: ast.Assign | ast.AnnAssign) -> ast.Name | None:
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            return None
        target = stmt.targets[0]
    else:
        target = stmt.target
    return target if isinstance(target, ast.Name) else None


def resolve_module_alias(stmt: ast.ImportFrom, cur_module: ModuleType) -> ModuleType:
    package_name = cur_module.__package__
    assert package_name is not None, "module is missing package metadata"
    if stmt.level > 0:
        module_name = importlib.util.resolve_name("." * stmt.level + (stmt.module or ""), package_name)
    else:
        assert stmt.module is not None, "import statement with no module"
        module_name = stmt.module
    return sys.modules.get(module_name) or importlib.import_module(module_name)


def bind_import_aliases(
    context: scoped_dict[str, Any],
    aliases: list[ast.alias],
    get_binding: Callable[[ast.alias], tuple[str, Any] | None],
) -> bool:
    for alias in aliases:
        binding = get_binding(alias)
        if binding is None:
            return False
        name, value = binding
        context[name] = value
    return True


def get_import_from_binding(module: ModuleType, alias: ast.alias) -> tuple[str, Any] | None:
    if alias.name == "*":
        return None
    try:
        value = getattr(module, alias.name)
    except AttributeError:
        value = importlib.import_module(f"{module.__name__}.{alias.name}")
    return alias.asname or alias.name, value


def get_import_binding(alias: ast.alias) -> tuple[str, ModuleType]:
    module = importlib.import_module(alias.name)
    bound_name = alias.asname or alias.name.split(".")[0]
    if alias.asname is None and "." in alias.name:
        module = sys.modules.get(bound_name) or importlib.import_module(bound_name)
    return bound_name, module


def bind_import_from_stmt(context: scoped_dict[str, Any], cur_module: ModuleType, stmt: ast.ImportFrom) -> bool:
    module = resolve_module_alias(stmt, cur_module)
    return bind_import_aliases(context, stmt.names, lambda alias: get_import_from_binding(module, alias))


def bind_import_stmt(context: scoped_dict[str, Any], stmt: ast.Import) -> bool:
    return bind_import_aliases(context, stmt.names, get_import_binding)


def resolve_name_ref(name: str, cur_module: ModuleType, filter: FilterFn) -> tuple[str, ModuleType]:
    # Bottom out at the leaf modules.
    if filter(cur_module):
        return name, cur_module
    source = inspect.getsource(cur_module)
    tree = ast.parse(source)
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.asname == name or (alias.asname is None and alias.name == name):
                    next_module = resolve_module_alias(stmt, cur_module)
                    return resolve_name_ref(alias.name, next_module, filter)
        elif isinstance(stmt, ast.Assign | ast.AnnAssign):
            target = get_assign_target(stmt)
            if target is not None and target.id == name:
                return name, cur_module
    raise ValueError(f"could not find module for {name} in {cur_module.__name__}")


def find_module(context: scoped_dict[str, Any], node: ast.AST) -> ModuleType | None:
    if isinstance(node, ast.Name):
        if node.id in context:
            module = context[node.id]
            if isinstance(module, ModuleType):
                return module
    elif isinstance(node, ast.Attribute):
        module = find_module(context, node.value)
        if module is not None:
            module = getattr(module, node.attr)
            if isinstance(module, ModuleType):
                return module
    return None


def get_reference(context: scoped_dict[str, Any], cur_module: ModuleType,
                  node: ast.AST) -> tuple[Any, ModuleType, str] | None:
    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load) or node.id not in context:
            return None
        return context[node.id], cur_module, node.id
    if isinstance(node, ast.Attribute):
        if not isinstance(node.ctx, ast.Load):
            return None
        rel_module = find_module(context, node.value)
        if rel_module is None:
            return None
        value = getattr(rel_module, node.attr)
        return value, rel_module, node.attr
    return None


def is_ignored_decorator(
    context: scoped_dict[str, Any],
    cur_module: ModuleType,
    decorator: ast.expr,
    rewrite_spec: RewriteSpec,
) -> bool:
    return any(matcher(context, cur_module, decorator) for matcher in rewrite_spec.ignored_decorator_matchers)


@dataclass
class Reference:
    value: GlobalValue
    module: ModuleType
    edges: ordered_set[int] | None = None
    mangled_name: str | None = None


@dataclass(frozen=True)
class LocalMarker:
    pass


@dataclass
class ReferenceScanner(ast.NodeVisitor):
    cur_module: ModuleType
    context: scoped_dict[str, Any]
    references: OrderedDict[int, Reference]
    queue: list[GlobalValue]
    value_remap: dict[int, GlobalValue]
    filter: FilterFn
    rewrite_spec: RewriteSpec = field(default_factory=RewriteSpec)

    edges: ordered_set[int] = field(default_factory=ordered_set[int])

    def process_reference(self, node: ast.Name | ast.Attribute, name: str, value: Any, rel_module: ModuleType) -> None:
        if isinstance(value, ModuleType | LocalMarker):
            return self.generic_visit(node)
        global_value = self.value_remap.get(id(value), None) or GlobalValue.wrap(
            value, name, lambda: resolve_name_ref(name, rel_module, self.filter))

        ref_id = global_value.id
        module = global_value.module
        if ref_id not in self.references:
            self.references[ref_id] = Reference(global_value, module)
            logger.debug(f"Added reference: {global_value} {module}")
            self.queue.append(global_value)

        # reference = self.references[ref_id]
        # assert reference.module is module, f"inconsistent value reference {global_value}"
        self.edges.add(ref_id)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return self.generic_visit(node)
        if node.id not in self.context:
            return self.generic_visit(node)
        value = self.context[node.id]
        return self.process_reference(node, node.id, value, self.cur_module)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not isinstance(node.ctx, ast.Load):
            return self.generic_visit(node)
        rel_module = find_module(self.context, node.value)
        if rel_module is None:
            return self.generic_visit(node)
        value = getattr(rel_module, node.attr)
        return self.process_reference(node, node.attr, value, rel_module)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        original_decorators = node.decorator_list
        node.decorator_list = [
            decorator for decorator in original_decorators if not is_ignored_decorator(
                self.context,
                self.cur_module,
                decorator,
                self.rewrite_spec,
            )
        ]
        args = node.args
        with self.context.scope():
            for arg in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
                if arg is not None:
                    self.context[arg.arg] = LocalMarker()
            return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        target = get_assign_target(node)
        if target is not None:
            self.context[target.id] = LocalMarker()
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        target = get_assign_target(node)
        if target is not None:
            self.context[target.id] = LocalMarker()
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if bind_import_from_stmt(self.context, self.cur_module, node):
            return None
        return self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        if bind_import_stmt(self.context, node):
            return None
        return self.generic_visit(node)


def match_regex(path: str) -> bool:
    identifier = r"[a-zA-Z_][a-zA-Z0-9_]*"
    pattern = rf"^{identifier}(\.{identifier})*:{identifier}$"
    return re.match(pattern, path) is not None


def get_base_value(path: str) -> GlobalValue:
    if not match_regex(path):
        raise ValueError(f"invalid Python object format: {path}")
    module_str, value_name = path.split(":")
    module = importlib.import_module(module_str)
    return GlobalValue.wrap(
        getattr(module, value_name),
        value_name,
        lambda: (value_name, module),
    )


def is_submodule(module: ModuleType, leaf_modules: list[str]) -> bool:
    return any(module.__name__ == leaf_module or module.__name__.startswith(f"{leaf_module}.")
               for leaf_module in leaf_modules)


def mangle_name(name: str, module: ModuleType, reference_names: set[str]) -> str:
    if name not in reference_names:
        return name
    for part in reversed(module.__name__.split(".")):
        name = f"{part}_{name}"
        if name not in reference_names:
            return name
    raise ValueError(f"FIXME: failed to mangle a unique name for {name} in {module.__name__}")


def parse_expr(expr_str: str) -> ast.expr:
    expr: ast.stmt = ast.parse(expr_str).body[0]
    assert isinstance(expr, ast.Expr)
    return expr.value


RewriteFn = Callable[[GlobalValue, ordered_set[str]], ast.AST | None]


def sugar_rewrite(module: str, alias: str) -> RewriteFn:

    def rewrite(global_value: GlobalValue, imports: ordered_set[str]) -> ast.AST | None:
        if module not in sys.modules:
            return None
        if not hasattr(sys.modules[module], global_value.name):
            return None
        if id(getattr(sys.modules[module], global_value.name)) == global_value.id:
            imports.add(f"import {module} as {alias}" if alias != module else f"import {module}")
            return ast.Attribute(value=ast.Name(id=alias, ctx=ast.Load()), attr=global_value.name, ctx=ast.Load())
        return None

    return rewrite


def add_sugar_rewrites(rewrites: list[RewriteFn], translate_to_gluon: bool) -> None:
    if translate_to_gluon:
        rewrites.append(sugar_rewrite("triton.experimental.gluon.language", "gl"))
        rewrites.append(sugar_rewrite("triton.experimental.gluon", "gluon"))
    else:
        rewrites.append(sugar_rewrite("triton.language", "tl"))
        rewrites.append(sugar_rewrite("triton", "triton"))

    def sugar_tensor_descriptor(global_value: GlobalValue, imports: ordered_set[str]) -> ast.AST | None:
        if global_value.original_value is TensorDescriptor:
            imports.add("from triton.tools.tensor_descriptor import TensorDescriptor")
            return ast.Name(id="TensorDescriptor", ctx=ast.Load())
        return None

    rewrites.append(sugar_tensor_descriptor)

    def sugar_set_allocator(global_value: GlobalValue, imports: ordered_set[str]) -> ast.AST | None:
        if global_value.original_value is triton.set_allocator:
            imports.add("import triton")
            return ast.Attribute(value=ast.Name(id="triton", ctx=ast.Load()), attr="set_allocator", ctx=ast.Load())
        return None

    rewrites.append(sugar_set_allocator)


@dataclass
class ReferenceRewriter(ast.NodeTransformer):
    cur_module: ModuleType
    context: scoped_dict[str, Any]
    references: OrderedDict[int, Reference]
    imports: ordered_set[str]
    filter: FilterFn
    value_remap: dict[int, GlobalValue]
    rewrite_spec: RewriteSpec = field(default_factory=RewriteSpec)

    rewrites: list[RewriteFn] = field(default_factory=list)

    def process_reference(self, node: ast.Name | ast.Attribute, name: str, value: Any,
                          rel_module: ModuleType) -> ast.AST:
        if isinstance(value, ModuleType | LocalMarker):
            return self.generic_visit(node)
        global_value = self.value_remap.get(id(value), None) or GlobalValue.wrap(
            value, name, lambda: resolve_name_ref(name, rel_module, self.filter))

        ref_id = global_value.id
        if ref_id not in self.references:
            return self.generic_visit(node)
        reference = self.references[ref_id]
        assert reference.mangled_name

        module = global_value.module
        name = global_value.name
        if self.filter(module) or self.filter(global_value):
            for rewrite in self.rewrites:
                result = rewrite(global_value, self.imports)
                if result is not None:
                    return result

            # Special rule for aliases to builtins.
            if module.__name__ == "builtins" and rel_module.__name__ != "builtins":
                self.imports.add(f"import {rel_module.__name__}")
                return node

            parts = module.__name__.split(".")
            reference_node: ast.Name | ast.Attribute = ast.Name(id=parts[0], ctx=ast.Load())
            for part in parts[1:]:
                reference_node = ast.Attribute(value=reference_node, attr=part, ctx=ast.Load())
            self.imports.add(f"import {module.__name__}")
            return ast.Attribute(value=reference_node, attr=name)

        return ast.Name(id=reference.mangled_name, ctx=node.ctx)

    def get_reference(self, node: ast.AST) -> tuple[Any, ModuleType, str] | None:
        return get_reference(self.context, self.cur_module, node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        ref = self.get_reference(node)
        if ref is None:
            return self.generic_visit(node)
        value, rel_module, name = ref
        return self.process_reference(node, name, value, rel_module)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        ref = self.get_reference(node)
        if ref is None:
            return self.generic_visit(node)
        value, rel_module, name = ref
        return self.process_reference(node, name, value, rel_module)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        for rewriter in self.rewrite_spec.annotation_rewriters:
            replacement = rewriter(self.context, self.cur_module, node)
            if replacement is not None:
                return ast.copy_location(self.visit(replacement), node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        args = node.args
        with self.context.scope():
            for arg in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
                if arg is not None:
                    self.context[arg.arg] = LocalMarker()
            return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        target = get_assign_target(node)
        if target is not None:
            self.context[target.id] = LocalMarker()
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        target = get_assign_target(node)
        if target is not None:
            self.context[target.id] = LocalMarker()
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        if bind_import_from_stmt(self.context, self.cur_module, node):
            return None
        return self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> ast.AST | None:
        if bind_import_stmt(self.context, node):
            return None
        return self.generic_visit(node)


@dataclass
class SliceRewriter(ReferenceRewriter):
    target: TranslatorTarget = field(kw_only=True)
    translate_to_gluon: bool = False
    inline_helpers: ordered_set[str] = field(default_factory=ordered_set[str])
    cvt_context: list[bool] = field(default_factory=lambda: [False])

    def __post_init__(self) -> None:
        # Special rules for sugaring imports.
        add_sugar_rewrites(self.rewrites, self.translate_to_gluon)

    def emit_reference_impl(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            ref = self.get_reference(node)
            if ref is None:
                return None
            value, _, _ = ref
            if isinstance(value, LocalMarker):
                return None
            return value
        elif isinstance(node, ast.Attribute):
            base = self.emit_reference_impl(node.value)
            if base is None:
                return None
            return getattr(base, node.attr)
        return None

    def emit_reference(self, node: ast.AST) -> Any:
        try:
            return self.emit_reference_impl(node)
        except RuntimeError as e:
            # HACK: Workaround triton.runtime.driver.active failing on CPU-only machines.
            if "0 active drivers" in str(e):
                return node
            raise e

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if not self.translate_to_gluon:
            return super().visit_Attribute(node)

        # Manual manipulation of JITFunctions.
        value = self.emit_reference(node)
        new_node = super().visit_Attribute(node)
        if value is JITFunction:
            self.imports.add("import triton.experimental.gluon._runtime as gluon_runtime")
            new_node = parse_expr("gluon_runtime.GluonJITFunction")
        elif value is tl.tensor_descriptor:
            self.imports.add(self.target.tensor_descriptor_import)
            new_node = ast.Name(id="tensor_descriptor", ctx=ast.Load())
        return new_node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not self.translate_to_gluon:
            return self.generic_visit(node)

        # Rewrite host code when translating to Gluon.
        callee = self.emit_reference(node.func)
        is_cvt = isinstance(node.func, ast.Name) and node.func.id == "convert_host_descriptor"
        self.cvt_context.append(is_cvt)
        new_node = self.generic_visit(node)
        self.cvt_context.pop()
        if callee in [TensorDescriptor, TensorDescriptor.from_tensor, create_ragged_descriptor
                      ] and not self.cvt_context[-1]:
            self.inline_helpers.add("convert_host_descriptor")
            new_node = parse_expr(f"convert_host_descriptor({ast.unparse(new_node)})")
        return new_node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        new_decorators: list[ast.expr] = []
        for decorator in node.decorator_list:
            # Decorators are applied bottom to top, so decorators above the
            # matched downstream kernel decorator will be applied after.
            if is_ignored_decorator(
                    self.context,
                    self.cur_module,
                    decorator,
                    self.rewrite_spec,
            ):
                new_decorators = []
                continue
            new_decorators.append(decorator)
        node.decorator_list = new_decorators
        return super().visit_FunctionDef(node)


def is_stdlib_module(module: ModuleType) -> bool:
    modspec = module.__spec__
    # Native modules don't have a spec. Always treat them as leaf modules.
    if modspec is None:
        return True
    assert modspec is not None, f"module {module.__name__} is missing spec metadata"
    origin = modspec.origin
    assert origin is not None, "module is missing origin"
    if origin in ["built-in", "frozen"]:
        return True

    origin_path = Path(origin)
    # A virtual environment can load packages from its base Python installation.
    # Those packages live below the base stdlib path, but are not included in the
    # active environment's purelib or platlib paths.
    if any(part in {"site-packages", "dist-packages"} for part in origin_path.parts):
        return False

    sys_paths = sysconfig.get_paths()
    site_package_paths = [Path(sys_paths[key]) for key in ("purelib", "platlib") if key in sys_paths]
    if any(origin_path.is_relative_to(site_path) for site_path in site_package_paths):
        return False

    stdlib_paths = [Path(sys_paths[key]) for key in ("stdlib", "platstdlib") if key in sys_paths]
    return any(origin_path.is_relative_to(stdlib_path) for stdlib_path in stdlib_paths)


def find_references(
    base_values: list[GlobalValue],
    filter: FilterFn,
    value_remap: dict[int, GlobalValue],
    rewrite_spec: RewriteSpec | None = None,
) -> tuple[OrderedDict[int, Reference], dict[int, ordered_set[int]]]:
    references: OrderedDict[int, Reference] = OrderedDict()
    queue: list[GlobalValue] = []
    graph: dict[int, ordered_set[int]] = {}
    rewrite_spec = rewrite_spec or RewriteSpec()

    for base_value in base_values:
        base_value = value_remap.get(base_value.id, base_value)
        queue.append(base_value)
        references[base_value.id] = Reference(base_value, base_value.module)

    while len(queue):
        value = queue.pop(0)
        if filter(value.module) or filter(value):
            graph[value.id] = ordered_set()
            continue
        logger.debug(f"Processing: {value.name}")
        logger.debug(f"Value: {value}")
        scanner = ReferenceScanner(
            value.module,
            scoped_dict(value.get_contextual_defs()),
            references,
            queue,
            value_remap,
            filter,
            rewrite_spec=rewrite_spec,
        )
        tree = value.parse_ast()
        scanner.visit(tree)
        graph[value.id] = scanner.edges

    return references, graph


def mangle_reference_names(references: OrderedDict[int, Reference], filter: FilterFn) -> None:
    reference_names: set[str] = set()
    for reference in references.values():
        name = reference.value.name
        module = reference.value.module
        if filter(module) or filter(reference.value):
            mangled_name = f"{module.__name__}.{name}"
        else:
            mangled_name = mangle_name(name, module, reference_names)
        reference_names.add(mangled_name)
        reference.mangled_name = mangled_name
        logger.debug(f"Value: {reference.value}")
        logger.debug(f"Module: {reference.module}")
        logger.debug(f"Name: {name} -> {mangled_name}")


def find_jit_functions(
    base_values: list[GlobalValue],
    filter: FilterFn,
    rewrite_spec: RewriteSpec | None = None,
) -> list[GlobalValue]:

    def new_filter(value: ModuleType | GlobalValue) -> bool:
        if isinstance(value, GlobalValue) and isinstance(value.original_value, JITFunction):
            return True
        return filter(value)

    references, _ = find_references(
        base_values,
        new_filter,
        value_remap={},
        rewrite_spec=rewrite_spec,
    )
    return [
        reference.value for reference in references.values() if isinstance(reference.value.original_value, JITFunction)
    ]


def load_module_from_file(name: str, path: str | Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert module is not None
    spec.loader.exec_module(module)
    return module


def slice_kernel(
    root_paths: list[str],
    leaf_modules: list[str],
    include_below: list[str] | None = None,
    leaf_paths: list[str] | None = None,
    translate_to_gluon: bool = False,
    rewrite_spec: RewriteSpec | None = None,
    *,
    target: TranslatorTarget,
) -> str:
    rewrite_spec = rewrite_spec or RewriteSpec()
    base_values: list[GlobalValue] = [get_base_value(root_path) for root_path in root_paths]
    base_value_ids: set[int] = set()
    for leaf_path in leaf_paths or []:
        base_value = get_base_value(leaf_path)
        base_value_ids.add(base_value.id)

    def filter(value: ModuleType | GlobalValue) -> bool:
        if isinstance(value, ModuleType):
            if is_stdlib_module(value):
                return True
            if is_submodule(value, leaf_modules):
                return not is_submodule(value, include_below or [])
            return False
        return value.id in base_value_ids

    value_remap: dict[int, GlobalValue] = {}
    if translate_to_gluon:
        # FIXME: Refactor code to avoid circular imports.
        from triton.tools.triton_to_gluon_translator.translator import translate_kernels

        jit_functions = find_jit_functions(
            base_values,
            filter,
            rewrite_spec=rewrite_spec,
        )
        jit_functions = [fn for fn in jit_functions if not fn.original_value.is_gluon()]
        converted_functions = translate_kernels(jit_functions, target=target)
        module_file = tempfile.NamedTemporaryFile(delete=False, prefix="translated_", suffix=".py")
        module_path = Path(module_file.name)
        module_path.write_text(converted_functions)
        module = load_module_from_file("converted_functions", module_path)
        for fn in jit_functions:
            gluon_fn = getattr(module, fn.name)
            assert isinstance(gluon_fn, JITFunction)
            value_remap[fn.id] = GlobalValue.wrap(
                gluon_fn,
                fn.name,
                lambda: (fn.name, module),
            )

    references, graph = find_references(
        base_values,
        filter,
        value_remap,
        rewrite_spec=rewrite_spec,
    )
    mangle_reference_names(references, filter)

    output = ""
    imports: ordered_set[str] = ordered_set()
    inline_helpers: ordered_set[str] = ordered_set()

    # Use a stable toposort to order the references. This is because global
    # values in the same module can form reference cycles, but we need to
    # generate them in the same order as they are in the original source in case
    # they have to be resolved in that particular order.
    ordered_ids = stable_toposort(graph)
    for ref_id in reversed(ordered_ids):
        reference = references[ref_id]
        name = reference.value.name
        if reference.mangled_name != name:
            logger.debug(f"Name mangled: {name} -> {reference.mangled_name}")
        if filter(reference.value.module) or filter(reference.value):
            continue
        tree = reference.value.parse_ast()
        context = reference.value.get_contextual_defs()
        rewriter = SliceRewriter(
            reference.value.module,
            scoped_dict(context),
            references,
            imports,
            filter,
            value_remap,
            rewrite_spec=rewrite_spec,
            translate_to_gluon=translate_to_gluon,
            inline_helpers=inline_helpers,
            target=target,
        )
        tree = rewriter.visit(tree)
        source = ast.unparse(tree)
        assert reference.mangled_name is not None
        source = reference.value.mangle_source(source, reference.mangled_name)
        output += source + "\n\n\n"
    output = "\n".join(imports) + "\n\n" + output

    if translate_to_gluon:
        # HACK: This updates the strings generated by `specialize`.
        output = output.replace("@triton.jit", "@gluon.jit")
        output = output.replace("tl.constexpr", "gl.constexpr")

    for helper in inline_helpers:
        output += inline_helper_defs[helper].strip() + "\n"

    return output


def slice_kernel_from_trace(
    kernel_path: str,
    trace: list[dict[str, list[str]]],
    translate_to_gluon: bool,
    extra_modules: dict[str, str],
    rewrite_spec: RewriteSpec | None = None,
    *,
    target: TranslatorTarget,
) -> str:
    module_remap: dict[str, str] = {}
    for name, path in extra_modules.items():
        load_module_from_file(name, path)
        module_remap[name] = Path(path).with_suffix("").stem

    leaf_paths: set[str] = set()
    root_paths: set[str] = {kernel_path}
    for entry in trace:
        leaf_paths.update(entry["type_names"])
        root_paths.update(entry["jit_fn_names"])

    # Remove obvious leaf paths.
    for leaf_path in list(leaf_paths):
        base_module = leaf_path.split(":")[0].split(".")[0]
        if base_module in ["triton", "torch"] or is_stdlib_module(importlib.import_module(base_module)):
            leaf_paths.remove(leaf_path)

    sliced = slice_kernel(
        root_paths=sorted(root_paths),
        leaf_modules=["triton", "torch", "ki.spo"],
        leaf_paths=sorted(leaf_paths),
        translate_to_gluon=translate_to_gluon,
        rewrite_spec=rewrite_spec,
        target=target,
    )

    fn_name = lambda path: path.split(":")[1]
    if len(root_paths) > 1:
        jit_fns = list(root_paths - {kernel_path})
        remap_lines = "\n".join(f"    '{fn_name(fn)}': {fn_name(fn)}," for fn in jit_fns)
        sliced += ("\n" + f"""
{fn_name(kernel_path)}.__jit_fn_remap__ = {{
{remap_lines}
}}""")

    return sliced


def main(
    root_paths: list[str],
    leaf_modules: list[str],
    include_below: list[str] | None = None,
    leaf_paths: list[str] | None = None,
    translate_to_gluon: bool = False,
    output_path: str = "/tmp/reference.py",
    *,
    target: TranslatorTarget,
) -> None:
    output = slice_kernel(
        root_paths=root_paths,
        leaf_modules=leaf_modules,
        include_below=include_below,
        leaf_paths=leaf_paths,
        translate_to_gluon=translate_to_gluon,
        target=target,
    )
    with open(output_path, "w") as f:
        f.write(output)


def _main_cli() -> None:
    parser = argparse.ArgumentParser(description="Slice Triton Python kernels into a standalone file.")
    parser.add_argument("root_paths", nargs="+", help="Root symbols to keep, in module.path:object format.")
    parser.add_argument("--leaf-module", dest="leaf_modules", action="append", default=[],
                        help="Module roots to treat as leaves. May be repeated.")
    parser.add_argument("--include-below", action="append", default=[],
                        help="Leaf modules that should still be traversed below. May be repeated.")
    parser.add_argument("--leaf-path", dest="leaf_paths", action="append", default=[],
                        help="Specific symbols to treat as leaves. May be repeated.")
    parser.add_argument("--translate-to-gluon", action="store_true",
                        help="Translate Triton JIT callables to Gluon while slicing.")
    parser.add_argument("--output-path", default="/tmp/reference.py", help="Path to write the sliced output.")
    parser.add_argument("--target", required=True, help="Target architecture (e.g. nvidia, gfx1250).")
    args = parser.parse_args()
    main(
        root_paths=args.root_paths,
        leaf_modules=args.leaf_modules,
        include_below=args.include_below or None,
        leaf_paths=args.leaf_paths or None,
        translate_to_gluon=args.translate_to_gluon,
        output_path=args.output_path,
        target=TranslatorTarget(args.target),
    )


if __name__ == "__main__":
    _main_cli()
<|endoftext|>from ..errors import TritonError
from typing import Optional


class InterpreterError(TritonError):

    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message

    def __str__(self) -> str:
        return self.error_message or ""


class OutOfResources(TritonError):

    def __init__(self, required, limit, name):
        self.required = required
        self.limit = limit
        self.name = name

    def __str__(self) -> str:
        return f"out of resource: {self.name}, Required: {self.required}, Hardware limit: {self.limit}. Reducing block sizes or `num_stages` may help."

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.required, self.limit, self.name))


class PTXASError(TritonError):

    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message

    def __str__(self) -> str:
        error_message = self.error_message or ""
        return f"PTXAS error: {error_message}"


class AutotunerError(TritonError):

    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message

    def __str__(self) -> str:
        error_message = self.error_message or ""
        return f"Autotuner error: {error_message}"
<|endoftext|>from __future__ import annotations, division
import ast
import copy
import hashlib
import inspect
import itertools
import threading
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Concatenate, Generic, Iterable, Optional, ParamSpec, TYPE_CHECKING, TypeVar, overload, Dict, Any, Tuple

from triton.backends import BaseBackend
from types import ModuleType
from .. import knobs
from .driver import driver
from . import _async_compile
from .._utils import find_paths_if, get_iterable_path, type_canonicalisation_dict, is_namedtuple
from .cache import get_cache_key
from triton._C.libtriton import get_cache_invalidating_env_vars, native_specialize_impl, ir

TRITON_MODULE = "triton.language"
GLUON_MODULE = "triton.experimental.gluon.language"

INDENT_PATTERN = re.compile(r"^(?P<indent>[ \t]*)def\s+\w+\s*\(", re.MULTILINE)

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
U = TypeVar("U")

# -----------------------------------------------------------------------------
# Dependencies Finder
# -----------------------------------------------------------------------------


class DependenciesFinder(ast.NodeVisitor):
    """
    This AST visitor is used to find dependencies of a JITFunction. This can
    be used to invalidate a JITFunction's hash when its source code -- or
    that of its dependencies -- changes.

    This visitor also keeps track of the global variables touched by the
    JITFunction.  When we launch the kernel, we check that these have the same
    values as they did when we ran this visitor.  If not, we raise an error (or
    otherwise we could recompile).
    """

    def __init__(self, name, globals, nonlocals, src) -> None:
        super().__init__()
        self.name = name
        self.hasher = hashlib.sha256(src.encode("utf-8"))

        # This function's __globals__ dict.
        self.globals = globals
        self.nonlocals = nonlocals

        # Python builtins that can be accessed from Triton kernels.
        self.supported_python_builtins = {
            'float',
            'getattr',
            'int',
            'isinstance',
            'len',
            'list',
            'max',
            'min',
            'print',
            'range',
        }
        self.supported_modules = {
            GLUON_MODULE,
            TRITON_MODULE,
            "copy",
            "math",
        }

        # used_global_vals tells us which global variables are used by this
        # function and all those it transitively calls, plus the values of those
        # variables when each function was initially run.  (That is, if A calls
        # C, and B calls C, then the values for C in used_global_vals will be
        # from the first time C was run, either by A or B.)
        #
        # Each function may have a different __globals__ dict, so the global
        # variable `foo` may actually have a different value in the different
        # functions.  Thus this map is actually
        #  (var_name, id(__globals__)) -> (var_value, __globals__).
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

        self.visiting_arg_default_value = False

    @property
    def ret(self):
        return self.hasher.hexdigest()

    def _update_hash(self, func):
        assert isinstance(func, JITCallable)
        func_key = func.cache_key
        # Merge our used_global_vals with those of the called function,
        # after checking that all overlapping values are consistent.
        for k in self.used_global_vals.keys() & func.used_global_vals.keys():
            var_name, _ = k
            v1, _ = self.used_global_vals[k]
            v2, _ = func.used_global_vals[k]
            if v1 != v2:
                raise RuntimeError(
                    f"Global variable {var_name} has value {v1} when compiling {self.name}, but inner kernel {func.__name__} has conflicting value {v2} from when it was first compiled.  This is not allowed."
                )
        self.used_global_vals.update(func.used_global_vals)
        # update hash
        func_key += str(getattr(func, "noinline", False))
        self.hasher.update(func_key.encode("utf-8"))

    def record_reference(self, val, var_dict=None, name=None):
        from ..language.core import constexpr
        # Only keep track of "interesting" global variables, that non-evil users
        # might change.  Don't consider functions, modules, builtins, etc.  This
        # helps keep the list of vars we have to check small.
        if val is None or type(val) is ModuleType:
            return

        if getattr(val, "__triton_aggregate__", False):
            self.hasher.update(str(val.__annotations__).encode("utf-8"))
            for attr in val.hash_attrs:
                self.record_reference(attr)
            return

        if getattr(val, "__triton_builtin__", False):
            return

        # Stubs that aren't real functions
        if getattr(val, "__module__", "") == "triton.language.extra.libdevice":
            return

        if isinstance(val, JITCallable):
            self._update_hash(val)
            return

        if callable(val) and not isinstance(val, type) and not isinstance(val, constexpr):
            raise RuntimeError(f"Unsupported function referenced: {val}")

        # Python default arguments are resolved only once, when the
        # function is defined.  So if you do `foo(a=A)` and the value of
        # A changes, foo will still use the old value of A.
        # It would be pretty evil if someone did `import x` and then
        # `x = blah`.
        if self.visiting_arg_default_value:
            return

        if var_dict is not None:
            self.used_global_vals[(name, id(var_dict))] = (copy.deepcopy(val), var_dict)
        return

    def visit_Name(self, node):
        if type(node.ctx) is ast.Store:
            return node.id

        if node.id in self.local_names:
            # The global name is hidden by the local name.
            return None

        def name_lookup(name):
            val = self.globals.get(name, None)
            if val is not None:
                return val, self.globals
            val = self.nonlocals.get(name, None)
            if val is not None:
                return val, self.nonlocals
            return None, None

        val, var_dict = name_lookup(node.id)
        if node.id in self.supported_python_builtins:
            return val

        self.record_reference(val, var_dict, node.id)
        return val

    def visit_Tuple(self, node):
        # We need to explicitly return the tuple values so that visit_Assign can
        # access them in the case of `a, b = ...`.
        return [self.visit(elt) for elt in node.elts]

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        while isinstance(lhs, ast.Attribute):
            lhs = self.visit(lhs.value)
        lhs_name = getattr(lhs, "__name__", "")
        if lhs is None or lhs_name in self.supported_modules:
            return None
        ret = getattr(lhs, node.attr, None)
        self.record_reference(ret)
        return ret

    def visit_FunctionDef(self, node):
        # Save the local name, which may hide the global name.
        self.local_names = {arg.arg for arg in node.args.args}
        self.generic_visit(node)

    def visit_arguments(self, node):
        # The purpose of this function is to visit everything in `arguments`
        # just like `generic_visit`, except when we're visiting default values
        # (i.e. the `foo` part of `def fn(x = foo)`), we set
        # self.visiting_arg_default_value = True.  This allows visit_Name to be
        # aware that we're inside function default values, which have special
        # semantics.

        # According to the AST docs, the arguments node has the following structure.
        #
        # arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
        #              expr* kw_defaults, arg? kwarg, expr* defaults)
        def visit_defaults(defaults):
            try:
                assert not self.visiting_arg_default_value
                self.visiting_arg_default_value = True
                for expr in defaults:
                    if expr is not None:
                        self.visit(expr)
            finally:
                self.visiting_arg_default_value = False

        for arg in itertools.chain(node.posonlyargs, node.args, [node.vararg] if node.vararg else [], node.kwonlyargs):
            self.visit(arg)

        visit_defaults(node.kw_defaults)

        if node.kwarg is not None:
            self.visit(node.kwarg)

        visit_defaults(node.defaults)

    def visitAssnTarget(self, node):
        # Target is either a single string, or a (possibly nested) list of strings if the assign target is a tuple.
        target = self.visit(node)

        def _add(t):
            if isinstance(t, list):
                for sub in t:
                    _add(sub)
            else:
                self.local_names.add(t)

        _add(target)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            # TODO(jlebar): I don't actually know how to hit this.  You don't
            # get it from `a, b = ...` -- in that case, node.targets is a single
            # Tuple, and in fact we *do* need to handle that case if we want
            # existing code to work.
            raise TypeError("Simultaneous multiple assignment is not supported.")

        self.visitAssnTarget(node.targets[0])

        # This will re-visit the target, but that's OK.
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.visitAssnTarget(node.target)

        # This will re-visit the target, but that's OK.
        self.generic_visit(node)

    def visit_For(self, node):
        self.visitAssnTarget(node.target)

        # This will re-visit the target, but that's fine.
        self.generic_visit(node)


# -----------------------------------------------------------------------------
# JITFunction
# -----------------------------------------------------------------------------


def _normalize_ty(ty) -> str:
    import triton.language.core as core
    if isinstance(ty, str):
        ty = ty.strip()
        if ty.startswith("const "):
            ty = ty.removeprefix("const")
            ty = _normalize_ty(ty)
            assert ty.startswith("*")
            return "*k" + ty[1:]
        if ty.endswith("*"):
            return "*" + _normalize_ty(ty[:-1])
        if ty.startswith("*"):
            return "*" + _normalize_ty(ty[1:])
        if ty.startswith("tl."):
            return _normalize_ty(ty.removeprefix("tl."))
    elif isinstance(ty, core.pointer_type):
        return f"*{_normalize_ty(ty.element_ty)}"
    elif isinstance(ty, core.dtype):
        ty = ty.name
    elif isinstance(ty, type):
        ty = ty.__name__
    else:
        ty = str(ty)
    return type_canonicalisation_dict.get(ty.replace("_t", ""), ty)


class KernelParam:
    """Represents a parameter (name plus metadata) to a @jit'ed function."""

    def __init__(self, num: int, param: inspect.Parameter, do_not_specialize: bool,
                 do_not_specialize_on_alignment: bool):
        self.num = num
        self._param = param
        self.do_not_specialize = do_not_specialize
        self.do_not_specialize_on_alignment = do_not_specialize_on_alignment

    @cached_property
    def name(self):
        return self._param.name

    @cached_property
    def annotation(self) -> str:
        if not self._param.annotation or self._param.annotation == inspect.Parameter.empty:
            return ""
        return _normalize_ty(self._param.annotation)

    @cached_property
    def annotation_type(self) -> str:
        a = self.annotation
        if a.startswith("*k"):
            a = a[2:]
        elif a.startswith("*"):
            a = a[1:]
        if a in set(type_canonicalisation_dict.values()):
            return self.annotation
        return ""

    @cached_property
    def is_constexpr(self):
        return "constexpr" in self.annotation

    @cached_property
    def is_const(self):
        if self.is_constexpr:
            return False
        return "const" in self.annotation or self.annotation.startswith("*k")

    @property
    def default(self):
        return self._param.default

    @property
    def has_default(self):
        return self._param.default != inspect.Parameter.empty


def mangle_type(arg, specialize=False):
    is_const = False
    align = True
    return native_specialize_impl(BaseBackend, arg, is_const, specialize, align)[0]


class KernelInterface(Generic[T]):
    run: T

    def warmup(self, *args, grid, **kwargs):
        return self.run(grid=grid, warmup=True, *map(MockTensor.wrap_dtype, args), **kwargs)

    def run(self, *args, grid, warmup, **kwargs):
        raise NotImplementedError("run not implemented")

    def __getitem__(self, grid) -> T:
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
        # return cast(T, functools.partial(cast(Callable, self.run), grid=grid))


def serialize_specialization_data(name, signature, constants, attrs, options, key, target):
    constants = {
        key: str(value) if value.__class__.__name__ == "dtype" else {"constexpr": value.value}
        if value.__class__.__name__ == "constexpr" else {"jit_function": f"{value.module}:{value.fn.__qualname__}"}
        if value.__class__.__name__ == "JITFunction" else value
        for key, value in constants.items()
    }

    import json
    obj = {
        'name': name, 'signature': signature, 'constant_keys': [list(x) for x in constants.keys()], 'constant_vals':
        list(constants.values()), 'attrs_keys': [list(x) for x in attrs.keys()], 'attrs_vals': list(attrs.values()),
        'options': options.__dict__, 'key': key, 'target': target.__dict__
    }
    serialized_obj = json.dumps(obj)
    return serialized_obj


def create_function_from_signature(sig, kparams, backend):
    """
    Equivalent to sig.bind followed by apply_defaults. This generates a
    native Python function (using exec) which can be memoized on a per-kernel
    basis to avoid having to run these expensive functions -- which constitute
    much of the kernel launch overhead -- every time we run the kernel.
    """
    assert len(sig.parameters) == len(kparams)
    # Create the function argument list and the dict entries for the return statement
    specialization = []
    # signature
    for name, kp in zip(sig.parameters.keys(), kparams):
        if kp.is_constexpr:
            specialization.append(f'("constexpr", {name})')
        else:
            is_const = 'True' if kp.is_const else 'False'
            specialize = 'False' if kp.do_not_specialize else 'True'
            align = 'False' if kp.do_not_specialize_on_alignment else 'True'
            ret = f"specialize_impl(backend, {name}, {is_const}, {specialize}, {align})"
            if kp.annotation_type:
                if isinstance(kp.annotation_type, str):
                    if kp.annotation_type == "u1" or kp.annotation_type[:2] in ["fp", "bf"]:
                        # we do not specialize non-constexpr floats and bools:
                        specialize = False
                if specialize:
                    specialization.append(f'("{kp.annotation_type}",) + {ret}[1:]')
                else:
                    # skip runtime specialization:
                    specialization.append(f'("{kp.annotation_type}", None)')
            else:
                specialization.append(f"{ret}")

    # compute argument string for a given parameter
    def arg(name_param):
        name, param = name_param
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return f"*{name}"
        return name if param.default is inspect.Parameter.empty else f"{name}=default_{name}"

    func_body = f"""
def dynamic_func({", ".join(list(map(arg, sig.parameters.items())) + ["**options"])}):
    params = {{{', '.join([f"'{name}': {name}" for name in sig.parameters.keys()])}}}
    specialization = [{','.join(specialization)}]
    return params, specialization, options
"""

    # Prepare defaults to be inserted into function namespace
    func_namespace = {
        f"default_{name}": param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    specialize_impl = native_specialize_impl
    func_namespace["specialize_impl"] = specialize_impl
    func_namespace["backend"] = backend
    func_namespace["JITCallable"] = JITCallable

    # Execute the function string in func_namespace to create the function
    exec(func_body, func_namespace)

    # Extract the newly created function from the namespace
    return func_namespace['dynamic_func']


def get_full_name(fn):
    return f"{fn.__module__}.{fn.__qualname__}"


class JITCallable:

    def __init__(self, fn):
        self.fn = fn
        self.signature = inspect.signature(fn)
        try:
            self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
        except OSError as e:
            raise ValueError("@jit functions should be defined in a Python file") from e
        self._fn_name = get_full_name(fn)
        self._hash_lock = threading.RLock()

        # function source code (without decorators)
        raw_src_str = "".join(self.raw_src)

        # get file name, starting line number and starting col number
        self.file_name = fn.__code__.co_filename
        self.def_file_line_number = get_def_line_number(self.raw_src, self.starting_line_number)
        self.def_file_col_number = get_def_col_number(raw_src_str)

        src = textwrap.dedent(raw_src_str)
        src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
        self._src = src
        self.hash = None

        # Map of global variables used by the function and any functions it
        # transitively calls, plus their values.  The values are collected when
        # the function is first compiled.  Then every time we run the function,
        # we check that the values of the globals match what's expected,
        # otherwise we raise an error.
        #
        # Different functions can have different __globals__ maps, so the map
        # key is actually (var name, id(__globals__)), and the map value is
        # (value, __globals__).
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

        # reuse docs of wrapped function
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

    def get_capture_scope(self):
        fn = self.fn
        if fn.__closure__ is None:
            return self.__globals__
        nonlocals = {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
        return self.__globals__ | nonlocals

    @property
    def cache_key(self) -> str:
        # TODO : hash should be attribute of `self`
        with self._hash_lock:
            if self.hash is not None:
                return self.hash
            # Set a placeholder hash to break recursion in case the function
            # transitively calls itself. The full hash is set after.
            self.hash = f"recursion:{self._fn_name}"
            nonlocals = inspect.getclosurevars(self.fn).nonlocals
            dependencies_finder = DependenciesFinder(name=self._fn_name, globals=self.__globals__, nonlocals=nonlocals,
                                                     src=self.src)
            dependencies_finder.visit(self.parse())
            self.hash = dependencies_finder.ret + str(self.starting_line_number)
            self.used_global_vals = dict(sorted(dependencies_finder.used_global_vals.items()))

            from triton.language.core import constexpr
            constexpr_globals = [(name, val)
                                 for (name, _), (val, _) in self.used_global_vals.items()
                                 if isinstance(val, constexpr)]
            constexpr_globals.sort(key=lambda item: (item[0], repr(item[1])))
            self.hash += str(constexpr_globals)
            self.hash = hashlib.sha256(self.hash.encode("utf-8")).hexdigest()
        return self.hash

    def __hash__(self):
        return hash(self.cache_key)

    # we do not parse `src` in the constructor because
    # the user might want to monkey-patch self.src dynamically.
    # Our unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self._src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    @property
    def type(self):
        from triton.language.core import constexpr_type
        return constexpr_type(self)

    def _flatten_ir(self, handles: list[ir.value]) -> None:
        pass

    def _unsafe_update_src(self, new_src):
        """
        The only method allowed to modify src.
        Bypasses the __setattr__ restriction by calling super().__setattr__ directly.

        Note that it is the callers responsibility to make sure any triton functions that call this function have the `.hash` value reset to None.
        """
        self.hash = None
        self._src = new_src

    def _set_src(self):
        raise AttributeError("Cannot set attribute 'src' directly. "
                             "Use '_unsafe_update_src()' and manually clear `.hash` of all callers"
                             "instead.")

    def _get_src(self):
        return self._src

    src = property(fget=_get_src, fset=_set_src)


_triton_jit_function_registry = {}


@dataclass
class JitFunctionInfo:
    module: ModuleType
    name: str
    jit_function: JITFunction


def compute_cache_key(kernel_key_cache, specialization, options):
    key = (tuple(specialization), str(options))
    cache_key = kernel_key_cache.get(key, None)
    if cache_key is not None:
        return cache_key

    # Replace JITCallable objects with their hash, so the cache key will change if the src is updated
    def replace_callables(obj):
        if isinstance(obj, list):
            return [replace_callables(arg) for arg in obj]
        elif is_namedtuple(obj):
            results = [replace_callables(arg) for arg in obj]
            return obj.__class__(*results)
        elif isinstance(obj, tuple):
            return tuple(replace_callables(arg) for arg in obj)
        elif isinstance(obj, JITCallable):
            return obj.cache_key
        return obj

    cache_key = str(replace_callables(specialization)) + str(options)
    kernel_key_cache[key] = cache_key
    return cache_key


def convert_to_tuple_if_list(item):
    # If the incoming item is a list, recursively iterate through it to convert all lists therein into tuples
    if not isinstance(item, list):
        return item

    # The value must be a list at this point
    for i, nested_value in enumerate(item):
        item[i] = convert_to_tuple_if_list(nested_value)

    return tuple(item)


class JITFunction(JITCallable, KernelInterface[T]):

    def is_gluon(self):
        return False

    def _call_hook(
        self,
        hook,
        key,
        signature,
        target,
        device,
        constants,
        options,
        configs,
        is_warmup,
    ) -> bool | None:
        if not hook:
            return None

        name = self.fn.__qualname__
        module = self.fn.__module__
        arg_reprs = ", ".join([f"{param.name}: {ty}" for param, ty in zip(self.params, key[1])])
        repr = f"{name}[num_warps={options.num_warps}, num_ctas={options.num_ctas}, num_stages={options.num_stages}, enable_fp_fusion={options.enable_fp_fusion}, launch_cooperative_grid={options.launch_cooperative_grid}]({arg_reprs})"
        full_name = get_full_name(self.fn)

        specialization_data = serialize_specialization_data(full_name, signature, constants, configs[0], options, key,
                                                            target)

        kwargs = {
            'signature': signature,
            'device': device,
            'constants': constants,
            'num_warps': options.num_warps,
            'num_ctas': options.num_ctas,
            'num_stages': options.num_stages,
            'enable_fp_fusion': options.enable_fp_fusion,
            'launch_cooperative_grid': options.launch_cooperative_grid,
            'extern_libs': options.extern_libs,
            'configs': configs,
            'specialization_data': specialization_data,
            'is_warmup': is_warmup,
        }

        return hook(
            key=key,
            repr=repr,
            fn=JitFunctionInfo(module, name, self),
            compile={"key": key, **kwargs},
            is_manual_warmup=is_warmup,
            already_compiled=False,
        )

    def add_pre_run_hook(self, hook):
        '''
        Add a hook that will be executed prior to the execution of run
        function with args and kwargs passed into the kernel
        '''
        assert callable(hook)
        self.pre_run_hooks.append(hook)

    def create_binder(self):
        """
        Precompute as much as possible.
        """
        from ..compiler import CompiledKernel, compile, ASTSource, make_backend
        target = driver.active.get_current_target()
        backend = make_backend(target)
        self.CompiledKernel = CompiledKernel
        self.compile = compile
        self.ASTSource = ASTSource
        binder = create_function_from_signature(self.signature, self.params, backend)
        return {}, {}, target, backend, binder

    def _pack_args(self, backend, kwargs, bound_args, specialization, options):
        # options
        options = backend.parse_options(kwargs)
        # signature
        sigkeys = [x.name for x in self.params]
        sigvals = [x[0] for x in specialization]
        signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
        # check arguments
        assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
        assert "device" not in kwargs, "device option is deprecated; current device will be used"
        assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"
        for k in kwargs:
            if k not in options.__dict__ and k not in sigkeys:
                raise KeyError("Keyword argument %s was specified but unrecognised" % k)
        # constexprs
        constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
        constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
        # attributes
        attrvals = ['' if x[0] == 'constexpr' else x[1] for x in specialization]
        attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
        attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

        return options, signature, constexprs, attrs

    def run(self, *args, grid, warmup, **kwargs):
        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
        kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.
        bound_args, specialization, options = binder(*args, **kwargs)

        # add a cache field to the kernel specializations for kernel specific
        # pass pipelines
        if knobs.runtime.add_stages_inspection_hook is not None:
            inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
            specialization.append(f'("custom_pipeline", {inspect_stages_hash})')

        key = compute_cache_key(kernel_key_cache, specialization, options)
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            options, signature, constexprs, attrs = self._pack_args(backend, kwargs, bound_args, specialization,
                                                                    options)

            kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
            if kernel is None:
                return None

        # Check that used global values have not changed.
        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel

    def repr(self, _):
        return self._fn_name if self._repr is None else self._repr(_)

    def __init__(self, fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None,
                 noinline=None, repr=None, launch_metadata=None):
        do_not_specialize = do_not_specialize if do_not_specialize else []
        do_not_specialize_on_alignment = do_not_specialize_on_alignment if do_not_specialize_on_alignment else []

        super().__init__(fn)
        self.module = fn.__module__
        self.version = version
        self.do_not_specialize = do_not_specialize
        self.do_not_specialize_on_alignment = do_not_specialize_on_alignment
        self._repr = repr
        self.launch_metadata = launch_metadata
        # Register for simple deserialization of JITFunction constants
        _triton_jit_function_registry[f"{self.module}:{self.fn.__qualname__}"] = self

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            dns = i in do_not_specialize or param.name in do_not_specialize
            dns_oa = i in do_not_specialize_on_alignment or param.name in do_not_specialize_on_alignment
            self.params.append(KernelParam(i, param, dns, dns_oa))

        # cache of just-in-time compiled kernels
        self.device_caches = defaultdict(self.create_binder)

        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel = None
        self.debug = debug
        self.noinline = noinline

        # TODO(jlebar): Remove uses of these fields outside this file, then
        # remove the fields here.
        self.arg_names = [p.name for p in self.params]
        self.constexprs = [p.num for p in self.params if p.is_constexpr]

        # Hooks that will be called prior to executing "run"
        self.pre_run_hooks = []

    def preload(self, specialization_data):
        import json
        import triton.language as tl
        device = driver.active.get_current_device()
        deserialized_obj = json.loads(specialization_data)
        if deserialized_obj['name'] != self._fn_name:
            raise RuntimeError(
                f"Specialization data is for {deserialized_obj['name']} but trying to preload for {self._fn_name}")
        constant_keys = map(tuple, deserialized_obj['constant_keys'])
        constant_vals = deserialized_obj['constant_vals']
        _, _, target, backend, _ = self.device_caches[device]
        deserialized_target = deserialized_obj['target']
        # TODO: we could support loading a kernel signature serialized on a different target however
        # currently options are target specific so we would need to change that.
        if target.__dict__ != deserialized_target:
            raise RuntimeError(f"Specialization data is for {deserialized_target} but trying to preload for {target}")

        def _decode_constant(value):
            if tl.dtype.is_dtype(value):
                return tl.dtype(value)
            if isinstance(value, dict):
                if 'constexpr' in value:
                    return tl.constexpr(convert_to_tuple_if_list(value['constexpr']))
                if 'jit_function' in value:
                    jf_key = value['jit_function']
                    if jf_key in _triton_jit_function_registry:
                        return _triton_jit_function_registry[jf_key]
                    raise RuntimeError(f"Unable to resolve JITFunction {jf_key} for preload")
            return convert_to_tuple_if_list(value)

        constexprs = {key: _decode_constant(value) for key, value in zip(constant_keys, constant_vals)}
        attrs_keys = map(tuple, deserialized_obj['attrs_keys'])
        attrs_vals = deserialized_obj['attrs_vals']
        attrs = dict(zip(attrs_keys, attrs_vals))
        # JSON serializes tuples as lists, so they need to be converted back;
        # This can be done unconditionally, since lists are not accepted in Triton kernel signatures.
        signature = {key: convert_to_tuple_if_list(value) for key, value in deserialized_obj['signature'].items()}
        options = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in deserialized_obj['options'].items()
        }
        key = deserialized_obj['key']
        options = backend.parse_options(options)
        return self._do_compile(
            key,
            signature,
            device,
            constexprs,
            options,
            attrs,
            warmup=True,
        )

    def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
        kernel_cache, _, target, backend, _ = self.device_caches[device]

        if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, target, device, constexprs, options, [attrs],
                           warmup):
            return None
        src = self.ASTSource(self, signature, constexprs, attrs)

        async_mode = _async_compile.active_mode.get()
        if async_mode is not None:

            env_vars = get_cache_invalidating_env_vars()
            cache_key = get_cache_key(src, backend, options, env_vars)

            def async_compile():
                return self.compile(src, target=target, options=options.__dict__, _env_vars=env_vars)

            def finalize_compile(kernel):
                kernel_cache[key] = kernel
                self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, target, device, constexprs,
                                options, [attrs], warmup)

            kernel = async_mode.submit(cache_key, async_compile, finalize_compile)
            kernel_cache[key] = kernel
        else:
            kernel = self.compile(src, target=target, options=options.__dict__)
            kernel_cache[key] = kernel
            self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, target, device, constexprs, options,
                            [attrs], warmup)
        return kernel

    def __call__(self: "JITFunction[Callable[P, R]]", *args: P.args, **kwargs: P.kwargs) -> R:
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")

    if TYPE_CHECKING:

        @overload
        def __get__(self, instance: None, owner: Optional[type] = None) -> "JITFunction[T]":
            ...

        @overload
        def __get__(self: "JITFunction[Callable[Concatenate[U, P], R]]", instance: Any,
                    owner: Optional[type] = None) -> Callable[P, R]:
            ...

        def __get__(self, instance, owner=None):
            ...

    def __repr__(self):
        return f"JITFunction({self.module}:{self.fn.__qualname__})"


# -----------------------------------------------------------------------------
# `jit` decorator
# -----------------------------------------------------------------------------


@overload
def jit(fn: T) -> JITFunction[T]:
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
) -> Callable[[T], JITFunction[T]]:
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
) -> KernelInterface[T]:
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
        if knobs.runtime.interpret:
            from .interpreter import InterpretedFunction
            return InterpretedFunction(fn, version=version, do_not_specialize=do_not_specialize,
                                       do_not_specialize_on_alignment=do_not_specialize_on_alignment, debug=debug,
                                       noinline=noinline, repr=repr, launch_metadata=launch_metadata)
        else:
            return JITFunction(
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


# -----------------------------------------------------------------------------
# Utilities for mocking tensors
# -----------------------------------------------------------------------------


class MockTensor:
    """
    Can be used in place of real tensors when calling:
        kernel.warmup(MockTensor(torch.float32), ...)
    """

    @staticmethod
    def wrap_dtype(arg):
        if arg.__class__.__name__ == "dtype" and arg.__module__ == "torch":
            return MockTensor(arg)
        return arg

    def __init__(self, dtype, shape=None):
        if shape is None:
            shape = [1]
        self.dtype = dtype
        self.shape = shape

    def stride(self):
        strides = [1]
        for size in self.shape[1:]:
            strides.append(strides[-1] * size)
        return tuple(reversed(strides))

    @staticmethod
    def data_ptr():
        return 0  # optimistically assumes multiple of 16

    @staticmethod
    def ptr_range():
        return 0  # optimistically assumes 32 bit pointer range


class TensorWrapper:

    def __init__(self, base, dtype):
        self.dtype = dtype
        self.base = base
        self.data = base.data
        self.device = base.device
        self.shape = self.base.shape

    def data_ptr(self):
        return self.base.data_ptr()

    def stride(self, *args):
        return self.base.stride(*args)

    def __str__(self) -> str:
        return f"TensorWrapper[{self.dtype}]({self.base})"

    def element_size(self):
        return self.base.element_size()

    def cpu(self):
        return TensorWrapper(self.base.cpu(), self.dtype)

    def copy_(self, other):
        self.base.copy_(other.base)

    def clone(self):
        return TensorWrapper(self.base.clone(), self.dtype)

    def to(self, device):
        return TensorWrapper(self.base.to(device), self.dtype)

    def new_empty(self, sizes):
        return TensorWrapper(self.base.new_empty(sizes), self.dtype)


def reinterpret(tensor, dtype):
    if isinstance(tensor, TensorWrapper):
        if dtype == tensor.base.dtype:
            # Reinterpreting to the original interpretation; return the base.
            return tensor.base
        else:
            # Reinterpreting a wrapped tensor to a different type.
            return TensorWrapper(tensor.base, dtype)
    elif hasattr(tensor, "data_ptr"):
        # A new wrapper is needed around an unwrapped tensor.
        return TensorWrapper(tensor, dtype)
    else:
        raise TypeError(f"Cannot reinterpret a {type(tensor)}.")


def get_def_line_number(raw_src, starting_line_number):
    def_file_line_number = starting_line_number
    # Match the following pattern:
    # @triton.autotune(...) <- foo.__code__.co_firstlineno
    # @triton.heuristics(...)
    # @triton.jit
    # def foo(...): <- this line is the first line
    for idx, line in enumerate(raw_src):
        if line.strip().startswith("def "):
            def_file_line_number += idx
            break
    return def_file_line_number


def get_def_col_number(raw_src_str):
    # Find the amount of indenting to use in the source location information.
    indented_def = INDENT_PATTERN.search(raw_src_str)
    if not indented_def:
        raise ValueError("No function definition found for kernel")
    # Consider spaces and tabs as single characters to match the ast
    def_file_col_number = len(indented_def.group("indent"))
    # Columns start at 1
    def_file_col_number += 1
    return def_file_col_number


class BoundConstexprFunction(JITCallable):

    def __init__(self, instance, fn):
        self.__self__ = instance
        self.__func__ = fn

    @property
    def cache_key(self):
        return self.__func__.cache_key

    def __call__(self, *args, **kwargs):
        return self.__func__(self.__self__, *args, **kwargs)


class ConstexprFunction(JITCallable, Generic[T]):

    def __init__(self, fn):
        super().__init__(fn)

    def __get__(self, obj, objclass):
        # Create a bound function to support constexpr_function methods
        if obj is not None:
            return BoundConstexprFunction(obj, self)
        return self

    @overload
    def __call__(self: "ConstexprFunction[Callable[P, R]]", *args: P.args, **kwargs: P.kwargs) -> R:
        ...

    def __call__(self, *args, _semantic=None, **kwargs):
        from triton.language.core import _unwrap_if_constexpr, constexpr
        # de-constexpr arguments and discard the _semantic keyword argument:
        args = [_unwrap_if_constexpr(x) for x in args]
        kwargs = {k: _unwrap_if_constexpr(v) for (k, v) in kwargs.items()}

        # call the raw Python function f:
        res = self.fn(*args, **kwargs)

        if _semantic is None:
            # Not called by triton code generator, e.g. in host code, another constexpr function, or even an aggreate's __init__ function
            return res

        # convert result back to a Triton constexpr:
        if knobs.runtime.interpret:
            return res  # No constexpr in interpreter
        return constexpr(res)


def constexpr_function(fn: T) -> ConstexprFunction[T]:
    """
    Wraps an arbitrary Python function so that it can be called at
    compile-time on constexpr arguments in a Triton function and
    returns a constexpr result.
    """
    return ConstexprFunction(fn)
<|endoftext|>from __future__ import annotations
import ast
import textwrap
import inspect
from typing import Tuple, List, Dict, Callable, TypeVar, Optional

import math
import numpy as np

import triton
import triton.language as tl
import dataclasses
from dataclasses import dataclass

from triton.language.semantic import TritonSemantic
from triton.runtime.jit import KernelInterface
from triton.tools.tensor_descriptor import TensorDescriptor
from .errors import InterpreterError
from functools import partial
from .._C.libtriton import interpreter as _interpreter  # type: ignore
from .._C.libtriton import ir as _ir  # type: ignore
from .._utils import _tuple_create

T = TypeVar("T")


@dataclass
class TensorHandle:
    '''
        data: numpy array
        dtype: triton type, either pointer_type or scalar_type.
        we don't store block_type here because the shape information is already available in the data field
        attr: a dictionary of attributes
    '''
    data: np.ndarray
    dtype: tl.dtype
    attr: Dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not _validate_np_data_size(self.data, self.dtype):
            raise ValueError(f"numpy data itemsize ({self.data.itemsize * 8} bits) exceeds dtype primitive_bitwidth "
                             f"({self.dtype.primitive_bitwidth} bits) for triton type {self.dtype}")

    def __bool__(self):
        return bool(self.data.all())

    def get_element_ty(self):
        dtype = self.dtype
        while hasattr(dtype, "element_ty"):
            dtype = dtype.element_ty
        return dtype

    def clone(self):
        return TensorHandle(self.data.copy(), self.dtype)

    def set_attr(self, key, value):
        self.attr[key] = value


class TensorDescHandle:

    def __init__(self, base: TensorHandle, shape: List[TensorHandle], strides: List[TensorHandle],
                 block_shape: List[int], padding):
        self.base = base
        self.ndim = len(shape)
        self.shape = shape
        self.strides = strides
        self.block_shape = block_shape
        self.padding = padding

    def validate(self):
        assert self.base.data.item() % 16 == 0, "base must be 16-byte aligned"
        assert len(self.strides) == self.ndim
        assert len(self.block_shape) == self.ndim
        assert self.ndim >= 1, "descriptor cannot be 0 dimensional"

        scalar_ty = self.base.dtype.element_ty
        itemsize = scalar_ty.primitive_bitwidth // 8
        for stride in self.strides[:-1]:
            byte_stride = stride.data.item() * itemsize
            assert byte_stride % 16 == 0, "stride must be 16-byte aligned"
        assert self.strides[-1].data.item() == 1, "last dim must be contiguous"

    def materialize_pointers(self, offsets: List[TensorHandle]):
        assert len(offsets) == self.ndim
        scalar_ty = self.base.dtype.element_ty
        itemsize = scalar_ty.primitive_bitwidth // 8
        assert (offsets[-1].data * itemsize) % 16 == 0, "block offset start must be 16-byte aligned"

        ptrs_data = np.broadcast_to(self.base.data, self.block_shape)
        masks = np.ones(self.block_shape, dtype=bool)
        for dim in range(len(self.block_shape)):
            bcast_dims = [1] * len(self.block_shape)
            bcast_dims[dim] = self.block_shape[dim]
            off = (offsets[dim].data + np.arange(self.block_shape[dim])).reshape(bcast_dims)
            ptrs_data = ptrs_data + (itemsize * off * self.strides[dim].data).astype(np.uint64)
            masks = masks & (0 <= off) & (off < self.shape[dim].data)
        assert ptrs_data.dtype == np.uint64
        ptrs_handle = TensorHandle(ptrs_data, self.base.dtype.scalar)
        return ptrs_handle, masks


@dataclass(frozen=True)
class InterpreterOptions:
    extern_libs: Optional[dict] = None
    debug: bool = False
    sanitize_overflow: bool = True
    arch: Optional[str] = None
    supported_fp8_dtypes: Tuple[str, ...] = ("fp8e5", "fp8e5b16", "fp8e4nv", "fp8e4b8", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str, ...] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str, ...] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: int = 0
    backend_name: str = "interpreter"


def _validate_np_data_size(np_array, tl_dtype):
    if isinstance(tl_dtype, tl.pointer_type):
        return True

    np_dtype_bitwidth = np_array.itemsize * 8
    tl_dtype_bitwidth = tl_dtype.primitive_bitwidth

    # numpy lowest itemsize is at least 8 bits
    if tl_dtype_bitwidth < 8:
        tl_dtype_bitwidth = 8

    if np_dtype_bitwidth > tl_dtype_bitwidth:
        return False
    return True


def _get_signed_np_dtype(dtype):
    if dtype == np.uint8:
        return np.int8
    if dtype == np.uint16:
        return np.int16
    if dtype == np.uint32:
        return np.int32
    if dtype == np.uint64:
        return np.int64
    return dtype


def _get_np_dtype(tt_dtype):
    if isinstance(tt_dtype, tl.pointer_type):
        return np.dtype(np.uint64)
    np_types = {
        tl.int1: np.dtype(bool),
        tl.float16: np.dtype(np.float16),
        tl.float32: np.dtype(np.float32),
        tl.float64: np.dtype(np.float64),
        tl.int8: np.dtype(np.int8),
        tl.uint8: np.dtype(np.uint8),
        tl.int16: np.dtype(np.int16),
        tl.uint16: np.dtype(np.uint16),
        tl.int32: np.dtype(np.int32),
        tl.uint32: np.dtype(np.uint32),
        tl.int64: np.dtype(np.int64),
        tl.uint64: np.dtype(np.uint64),
        # bfloat16 types are stored as uint16
        tl.bfloat16: np.dtype(np.uint16),
        # float8 types are stored as uint8
        tl.float8e5: np.dtype(np.uint8),
        tl.float8e5b16: np.dtype(np.uint8),
        tl.float8e4nv: np.dtype(np.uint8),
        tl.float8e4b8: np.dtype(np.uint8),
        tl.float8e4b15: np.dtype(np.uint8),
    }
    if isinstance(tt_dtype, tl.block_type):
        if isinstance(tt_dtype.element_ty, tl.pointer_type):
            return np.dtype(np.uint64)
        return np_types[tt_dtype.element_ty]
    return np_types[tt_dtype]


def _convert_float(input, input_dtype, output_dtype, rounding_mode):
    input_uint_dtype = getattr(np, f"uint{input_dtype.primitive_bitwidth}")
    output_unint_dtype = getattr(np, f"uint{output_dtype.primitive_bitwidth}")
    input_bin = np.frombuffer(input.tobytes(), dtype=input_uint_dtype)
    sign = (input_bin >> (input_dtype.primitive_bitwidth - 1)) & 0x01
    input_exponent_width = input_dtype.primitive_bitwidth - input_dtype.fp_mantissa_width - 1
    output_exponent_width = output_dtype.primitive_bitwidth - output_dtype.fp_mantissa_width - 1
    significand = input_bin & ((1 << input_dtype.fp_mantissa_width) - 1)
    bias_input = input_dtype.exponent_bias
    bias_output = output_dtype.exponent_bias
    exponent = ((input_bin >> input_dtype.fp_mantissa_width) & ((1 << input_exponent_width) - 1)).astype(np.int32)
    subnormal_index = exponent == 0
    if np.any(subnormal_index):
        # Credit to Phil: phil@openai.com
        # subnormal repr: ((-1.0)**sign) * (2.0**(1 - exp_bias)) * (2^(m0) + 2^(m1) + ... + 2^(mn))
        # where m0, m1, ..., mn are the 1-bit of the mantissa
        # convert it to normal repr: ((-1.0)**sign) * (2.0**(1 + m0 - exp_bias)) * (1 + 2^(m1 - m0) + ... + 2^(mn - m0))
        bit_pos = np.zeros_like(input_bin, dtype=np.int32)
        # Find the most significant bit of the mantissa in the significand
        for i in range(input_dtype.fp_mantissa_width):
            bit_index = ((significand >> i) & 0x01)
            # pos should be >= 1
            bit_pos[bit_index == 1] = input_dtype.fp_mantissa_width - i
        zero_significand_index = significand == 0
        exponent[subnormal_index] = 1 - bit_pos[subnormal_index]
        # 0 significand and subnormal should be treated as 0
        exponent[zero_significand_index & subnormal_index] = bias_input - bias_output
        significand[subnormal_index] = (significand[subnormal_index] << bit_pos[subnormal_index]) & (
            (1 << input_dtype.fp_mantissa_width) - 1)
    # Prevent overflow and underflow
    exponent_output = np.maximum(0, np.minimum((exponent - bias_input + bias_output), (1 << output_exponent_width) - 1))
    exponent_output = exponent_output.astype(output_unint_dtype)
    sign_output = sign.astype(output_unint_dtype)
    if input_dtype.primitive_bitwidth > output_dtype.primitive_bitwidth:  # Downcast
        significand_output = (significand >> (input_dtype.fp_mantissa_width - output_dtype.fp_mantissa_width)) & (
            (1 << output_dtype.fp_mantissa_width) - 1)
        if rounding_mode == _ir.ROUNDING_MODE.RTNE:  # Round to nearst even
            # find the cut-off bit
            cut_off = significand & (1 << (input_dtype.fp_mantissa_width - output_dtype.fp_mantissa_width - 1))
            significand_output = significand_output + (cut_off > 0)
        significand_output = significand_output.astype(output_unint_dtype)
    else:  # Upcast
        significand_output = (significand.astype(output_unint_dtype) <<
                              (output_dtype.fp_mantissa_width - input_dtype.fp_mantissa_width)) & (
                                  (1 << output_dtype.fp_mantissa_width) - 1)
    subnormal_index = exponent_output == 0
    if np.any(subnormal_index):  # underflow
        # normal repr: ((-1.0)**sign) * (2.0**(exp - exp_bias_input)) * (1 + 2^(m0) + 2^(m1) + ... + 2^(mn))
        # where m0, m1, ..., mn are the 1-bit of the mantissa
        # shift = (1 - exp_bias_output) - (exp - exp_bias_input)
        # convert it to subnormal repr: ((-1.0)**sign) * (2.0**(1 - exp_bias_output)) * (2^(-shift) + 2^(m0 - shift) + 2^(m1 - shift) + ... + 2^(mn - shift))
        exponent = ((input_bin >> input_dtype.fp_mantissa_width) & ((1 << input_exponent_width) - 1)).astype(np.int32)
        non_zero_exponent_index = exponent != 0
        # If the original exponent is not zero, we still need to shift the significand and consider the 1.0 part in mantissa
        subnormal_index = subnormal_index & non_zero_exponent_index
        shift = np.zeros_like(input_bin, dtype=np.int32)
        shift[subnormal_index] = (1 - bias_output) - (exponent[subnormal_index] - bias_input)
        significand_output[subnormal_index] = (significand_output[subnormal_index] >> shift[subnormal_index]) | (
            1 << (output_dtype.fp_mantissa_width - shift[subnormal_index]))
    output = (sign_output << (output_dtype.primitive_bitwidth - 1)) | (
        exponent_output << output_dtype.fp_mantissa_width) | significand_output
    return output.reshape(input.shape)


def _erf(x):
    # Numpy does not support erf
    return math.erf(x)


def _umulhi_64(a, b):
    # Numpy does not support 128-bit multiplication
    # So we have to implement it manually
    return (int(a) * int(b)) >> 64


def _e8m0_to_f32(scale):
    assert scale.dtype in (np.uint8, np.int8)
    scale = scale.astype(np.uint8)
    scale = scale.astype(np.int32)
    scale = scale << 23
    scale = scale.view(np.float32)
    return scale


def _e2m1_to_f32(value):
    assert value.dtype == np.uint8

    low = value & np.uint8(0x0F)
    high = value >> np.uint8(4)

    unpacked_shape = value.shape[:-1] + (value.shape[-1] * 2, )
    unpacked_val = np.empty(unpacked_shape, dtype=np.uint8)
    unpacked_val[..., 0::2] = low
    unpacked_val[..., 1::2] = high

    # 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6 (from Onnx)
    positive_e2m1_lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    abs_values = positive_e2m1_lut[(unpacked_val & np.uint8(0x07))]
    signs = (unpacked_val & np.uint8(0x08)) != 0
    return np.where(signs, -abs_values, abs_values)


def _mxfp_value_handle_to_float32(value_handle):
    if value_handle.dtype == tl.uint8:
        value_float = _e2m1_to_f32(value_handle.data)
    else:
        # Decode fp8 values through float16 first, then widen to float32.
        # This preserves the same intermediate rounding used by the reference
        # path, which is especially visible for float8e5m2.
        value_float = _convert_float(
            value_handle.data,
            value_handle.dtype,
            tl.float16,
            _ir.ROUNDING_MODE.RTNE,
        ).view(np.float16).astype(np.float32)

    # Handle inf/nan for e5m2/e4m3
    if value_handle.dtype == tl.float8e5:
        pos_inf_mask = value_handle.data == np.uint8(0x7C)
        value_float = np.where(pos_inf_mask, np.float32('inf'), value_float)

        neg_inf_mask = value_handle.data == np.uint8(0xFC)
        value_float = np.where(neg_inf_mask, -np.float32('inf'), value_float)

        nan_mask = np.logical_and((value_handle.data & np.uint8(0x7C)) == np.uint8(0x7C),
                                  (value_handle.data & np.uint8(3)) != np.uint8(0))
        value_float = np.where(nan_mask, np.float32('nan'), value_float)
    elif value_handle.dtype == tl.float8e4nv:
        nan_mask = value_handle.data & np.uint8(0x7F) == np.uint8(0x7F)
        value_float = np.where(nan_mask, np.float32('nan'), value_float)

    return value_float


def _unpack_e2m1(data, axis):
    # E2M1 stores two logical 4-bit values per byte; unpack on the physical
    # axis that carries the packed logical matrix dimension.
    data = np.moveaxis(data, axis, -1)
    unpacked = _e2m1_to_f32(data)
    return np.moveaxis(unpacked, -1, axis)


def _prepare_dot_scaled_operand(value_handle, scale_handle, format_enum, k_pack, is_rhs):
    if format_enum == _ir.ScaleDotElemTypeTY.E2M1:
        if is_rhs:
            unpack_axis = -2 if k_pack else -1
        else:
            unpack_axis = -1 if k_pack else -2
        value = _unpack_e2m1(value_handle.data, unpack_axis)
    else:
        value = _mxfp_value_handle_to_float32(value_handle)

    if scale_handle is None:
        return value

    scale = _e8m0_to_f32(scale_handle.data)

    if is_rhs:
        # rhs is in [K, N] layout, but rhs_scale is supplied as [N, K / group].
        scale = np.repeat(scale, value.shape[-2] // scale.shape[-1], axis=-1)
        scale = np.swapaxes(scale, -1, -2)
    else:
        scale = np.repeat(scale, value.shape[-1] // scale.shape[-1], axis=-1)

    return value * scale


np_erf_fp32 = np.vectorize(_erf, otypes=[np.float32])
np_erf_fp64 = np.vectorize(_erf, otypes=[np.float64])
np_umulhi_u64 = np.vectorize(_umulhi_64, otypes=[np.uint64])


class ExtraFunctions:

    @staticmethod
    def _convert_custom_types(input, dst_ty, fp_downcast_rounding, _semantic):
        return tl.tensor(_semantic.builder.create_fp_to_fp(input.handle, dst_ty, fp_downcast_rounding), dst_ty)


class InterpreterBuilder:
    ir_sem_to_interpreter_sem = {
        _ir.MEM_SEMANTIC.ACQUIRE: _interpreter.MEM_SEMANTIC.ACQUIRE,
        _ir.MEM_SEMANTIC.RELEASE: _interpreter.MEM_SEMANTIC.RELEASE,
        _ir.MEM_SEMANTIC.RELAXED: _interpreter.MEM_SEMANTIC.RELAXED,
        _ir.MEM_SEMANTIC.ACQUIRE_RELEASE: _interpreter.MEM_SEMANTIC.ACQUIRE_RELEASE,
    }

    ir_rmw_op_to_interpreter_rmw_op = {
        _ir.ATOMIC_OP.ADD: _interpreter.RMW_OP.ADD,
        _ir.ATOMIC_OP.FADD: _interpreter.RMW_OP.FADD,
        _ir.ATOMIC_OP.MIN: _interpreter.RMW_OP.MIN,
        _ir.ATOMIC_OP.UMIN: _interpreter.RMW_OP.UMIN,
        _ir.ATOMIC_OP.MAX: _interpreter.RMW_OP.MAX,
        _ir.ATOMIC_OP.UMAX: _interpreter.RMW_OP.UMAX,
        _ir.ATOMIC_OP.AND: _interpreter.RMW_OP.AND,
        _ir.ATOMIC_OP.OR: _interpreter.RMW_OP.OR,
        _ir.ATOMIC_OP.XOR: _interpreter.RMW_OP.XOR,
        _ir.ATOMIC_OP.XCHG: _interpreter.RMW_OP.XCHG,
    }

    def __init__(self) -> None:
        self.arch = None
        self.options = InterpreterOptions()
        self.codegen_fns = {}
        self.codegen_fns["convert_custom_types"] = ExtraFunctions._convert_custom_types
        self.codegen_fns["min_dot_size"] = lambda lhsType, rhsType: (1, 1, 1)

    def set_grid_idx(self, x, y, z):
        if not x < self.grid_dim[0]:
            raise ValueError("x >= grid_dim[0]")
        if not y < self.grid_dim[1]:
            raise ValueError("y >= grid_dim[1]")
        if not z < self.grid_dim[2]:
            raise ValueError("z >= grid_dim[2]")
        self.grid_idx = (x, y, z)

    def set_grid_dim(self, nx, ny, nz):
        self.grid_dim = (nx, ny, nz)

    # constants

    def get_half_ty(self):
        return tl.float16

    def get_bf16_ty(self):
        return tl.bfloat16

    def get_float_ty(self):
        return tl.float32

    def get_double_ty(self):
        return tl.float64

    def get_int1_ty(self):
        return tl.int1

    def get_int8_ty(self):
        return tl.int8

    def get_uint8_ty(self):
        return tl.uint8

    def get_int16_ty(self):
        return tl.int16

    def get_uint16_ty(self):
        return tl.uint16

    def get_int32_ty(self):
        return tl.int32

    def get_uint32_ty(self):
        return tl.uint32

    def get_int64_ty(self):
        return tl.int64

    def get_uint64_ty(self):
        return tl.uint64

    def get_fp8e4nv_ty(self):
        return tl.float8e4nv

    def get_fp8e4b15_ty(self):
        return tl.float8e4b15

    def get_fp8e4b8_ty(self):
        return tl.float8e4b8

    def get_fp8e5_ty(self):
        return tl.float8e5

    def get_fp8e5b16_ty(self):
        return tl.float8e5b16

    def get_ptr_ty(self, elt_ty, addr_space):
        return tl.pointer_type(elt_ty, addr_space)

    def get_block_ty(self, dtype, shape):
        return tl.block_type(dtype, shape)

    def get_int1(self, value):
        return TensorHandle(np.array([value], dtype=np.bool_), tl.int1)

    def get_uint8(self, value):
        return TensorHandle(np.array([value], dtype=np.uint8), tl.uint8)

    def get_int8(self, value):
        return TensorHandle(np.array([value], dtype=np.int8), tl.int8)

    def get_uint16(self, value):
        return TensorHandle(np.array([value], dtype=np.uint16), tl.uint16)

    def get_int16(self, value):
        return TensorHandle(np.array([value], dtype=np.int16), tl.int16)

    def get_uint32(self, value):
        return TensorHandle(np.array([value], dtype=np.uint32), tl.uint32)

    def get_int32(self, value):
        return TensorHandle(np.array([value], dtype=np.int32), tl.int32)

    def get_uint64(self, value):
        return TensorHandle(np.array([value], dtype=np.uint64), tl.uint64)

    def get_int64(self, value):
        return TensorHandle(np.array([value], dtype=np.int64), tl.int64)

    def get_fp16(self, value):
        return TensorHandle(np.array([value], dtype=np.float16), tl.float16)

    def get_fp32(self, value):
        return TensorHandle(np.array([value], dtype=np.float32), tl.float32)

    def get_fp64(self, value):
        return TensorHandle(np.array([value], dtype=np.float64), tl.float64)

    def get_null_value(self, type):
        return TensorHandle(np.array([0], dtype=_get_np_dtype(type)), type)

    # programming model
    def create_get_program_id(self, axis):
        if self.grid_idx is None:
            raise ValueError("grid_idx is None")
        return TensorHandle(np.array([self.grid_idx[axis]], dtype=np.int32), tl.int32)

    def create_get_num_programs(self, axis):
        return TensorHandle(np.array([self.grid_dim[axis]], dtype=np.int32), tl.int32)

    # memory ops
    def create_load(self, ptr, _0, _1, is_volatile):
        mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
        other = None
        return self.create_masked_load(ptr, mask, other, _0, _1, is_volatile)

    def create_store(self, ptr, val, _0, _1):
        mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
        return self.create_masked_store(ptr, val, mask, None, None)

    def create_masked_load(self, ptrs, mask, other, cache_modifier, eviction_policy, is_volatile):
        dtype_tt = ptrs.get_element_ty()
        dtype_np = _get_np_dtype(dtype_tt)
        if other is None:
            other = TensorHandle(np.zeros_like(ptrs.data, dtype=dtype_np), dtype_tt)
        ret = _interpreter.load(ptrs.data, mask.data, other.data, dtype_np)
        return TensorHandle(ret, dtype_tt)

    def create_masked_store(self, ptrs, value, mask, cache_modifier, eviction_policy):
        return _interpreter.store(ptrs.data, value.data, mask.data)

    # casting ops
    def cast_impl(self, src, dst_type):
        src_element_type = src.dtype.scalar
        dst_element_type = dst_type.scalar
        if (src_element_type == tl.bfloat16 and dst_element_type == tl.float32) or \
           (src_element_type == tl.float32 and dst_element_type == tl.bfloat16):
            data = _convert_float(src.data, src_element_type, dst_element_type, None).view(_get_np_dtype(dst_type))
            return TensorHandle(data, dst_type.scalar)
        else:
            return TensorHandle(src.data.astype(_get_np_dtype(dst_type)), dst_type.scalar)

    create_si_to_fp = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_ui_to_fp = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_to_si = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_to_ui = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_ext = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_trunc = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_int_cast = lambda self, src, dst_type, is_signed: self.cast_impl(src, dst_type)

    def create_fp_to_fp(self, src, dst_type, rounding_mode):
        src_element_type = src.dtype.scalar
        dst_element_type = dst_type.scalar
        data = _convert_float(src.data, src_element_type, dst_element_type, rounding_mode).view(_get_np_dtype(dst_type))
        return TensorHandle(data, dst_type.scalar)

    def create_bitcast(self, src, dst_type):
        return TensorHandle(src.data.view(_get_np_dtype(dst_type)), dst_type.scalar)

    # binary operators
    def binary_op(self, lhs, rhs, op):
        output = op(lhs.data, rhs.data)
        tl_dtype = lhs.dtype.scalar

        if not _validate_np_data_size(output, tl_dtype):
            output = output.astype(_get_np_dtype(tl_dtype))

        return TensorHandle(output, tl_dtype)

    create_fadd = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.add)
    create_fmul = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.multiply)
    create_fdiv = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.divide)
    create_frem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.fmod)
    create_fsub = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.subtract)
    create_fneg = lambda self, input: self.unary_op(input, np.negative)
    create_mul = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.multiply)
    create_precise_divf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.divide)
    create_sdiv = lambda self, lhs, rhs: self.create_idiv(lhs, rhs)
    create_udiv = lambda self, lhs, rhs: self.create_idiv(lhs, rhs)
    # LLVM has 'numpy.fmod', not 'numpy.remainder', semantics on integer remainders.
    create_srem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.fmod)
    create_urem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.fmod)
    create_add = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.add)
    create_sub = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.subtract)
    create_shl = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.left_shift)
    create_lshr = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.right_shift)
    create_minsi = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_minui = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_minimumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_minnumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.fmin)
    create_maxsi = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_maxui = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_maximumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_maxnumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.fmax)
    create_icmpSLE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_icmpSLT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_icmpSGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_icmpSGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_icmpULE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_icmpULT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_icmpUGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_icmpUGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_icmpEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_icmpNE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_fcmpOLT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_fcmpOGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_fcmpOLE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_fcmpOGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_fcmpOEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_fcmpONE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_fcmpULT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_fcmpUGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_fcmpULE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_fcmpUGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_fcmpUEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_fcmpUNE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_and = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_and)
    create_xor = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_xor)
    create_or = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_or)
    create_int_to_ptr = create_bitcast
    create_ptr_to_int = create_bitcast

    def create_idiv(self, lhs, rhs):
        # Triton has IEEE, not numpy/torch, semantics for %, and those carry
        # through to //, so we have to use a nonstandard expression to get a
        # reference result for //.
        return TensorHandle((lhs.data - np.fmod(lhs.data, rhs.data)) // rhs.data, lhs.dtype.scalar)

    def create_ashr(self, lhs, rhs):
        # Triton's rshift operator depends on the signedness of the left operand
        lhs_dtype = _get_signed_np_dtype(lhs.data.dtype)
        rhs_dtype = _get_signed_np_dtype(rhs.data.dtype)
        lhs_data = lhs.data.astype(lhs_dtype)
        rhs_data = rhs.data.astype(rhs_dtype)
        lhs = TensorHandle(lhs_data, lhs.dtype)
        rhs = TensorHandle(rhs_data, rhs.dtype)
        return self.binary_op(lhs, rhs, np.right_shift)

    def create_umulhi(self, lhs, rhs):
        dtype = lhs.data.dtype
        if dtype == np.int64 or dtype == np.uint64:
            return TensorHandle(np_umulhi_u64(lhs.data, rhs.data), lhs.dtype.scalar)
        else:
            compute_dtype = getattr(np, f"uint{dtype.itemsize * 8 * 2}")
            lhs_data = lhs.data.astype(compute_dtype)
            rhs_data = rhs.data.astype(compute_dtype)
            ret_data = np.multiply(lhs_data, rhs_data) >> (dtype.itemsize * 8)
            return TensorHandle(ret_data.astype(dtype), lhs.dtype.scalar)

    # ternary functions
    def ternary_op(self, lhs, rhs, other, op):
        output = op(lhs.data, rhs.data, other.data)
        tl_dtype = other.dtype.scalar

        if not _validate_np_data_size(output, tl_dtype):
            output = output.astype(_get_np_dtype(tl_dtype))

        return TensorHandle(output, tl_dtype)

    def create_clampf(self, arg, lo, hi, propagate_nans):
        if propagate_nans == _ir.PROPAGATE_NAN.NONE:
            return self.binary_op(self.binary_op(arg, lo, np.fmax), hi, np.fmin)
        else:
            return self.ternary_op(arg, lo, hi, np.clip)

    create_select = lambda self, cond, lhs, rhs: self.ternary_op(cond, lhs, rhs, np.where)

    def create_fma(self, x, y, z):
        return TensorHandle(x.data * y.data + z.data, z.dtype.scalar)

    # unary functions
    def unary_op(self, arg, op):
        return TensorHandle(op(arg.data), arg.dtype.scalar)

    def create_fabs(self, arg):
        # Mask out the sign bit based on the primitive length
        dtype_tt = arg.dtype
        mask_bitwidth = dtype_tt.primitive_bitwidth - 1
        np_uint_dtype = getattr(np, f"uint{dtype_tt.primitive_bitwidth}")
        data = arg.data.view(np_uint_dtype)
        mask = (1 << mask_bitwidth) - 1
        ret = (data & mask).view(_get_np_dtype(dtype_tt))
        return TensorHandle(ret, arg.dtype.scalar)

    create_cos = lambda self, arg: self.unary_op(arg, np.cos)
    create_exp = lambda self, arg: self.unary_op(arg, np.exp)
    create_exp2 = lambda self, arg: self.unary_op(arg, np.exp2)
    create_iabs = lambda self, arg: self.unary_op(arg, np.abs)
    create_floor = lambda self, arg: self.unary_op(arg, np.floor)
    create_ceil = lambda self, arg: self.unary_op(arg, np.ceil)
    create_log = lambda self, arg: self.unary_op(arg, np.log)
    create_log2 = lambda self, arg: self.unary_op(arg, np.log2)
    create_precise_sqrt = lambda self, arg: self.unary_op(arg, np.sqrt)
    create_sqrt = lambda self, arg: self.unary_op(arg, np.sqrt)
    create_sin = lambda self, arg: self.unary_op(arg, np.sin)

    def create_erf(self, arg):
        ret = np_erf_fp32(arg.data) if arg.data.dtype == np.float32 else np_erf_fp64(arg.data)
        return TensorHandle(ret, arg.dtype.scalar)

    def create_rsqrt(self, arg):
        return TensorHandle(1 / np.sqrt(arg.data), arg.dtype.scalar)

    # tensor operators
    create_reshape = lambda self, arg, shape, allow_reorder: TensorHandle(arg.data.reshape(shape), arg.dtype.scalar)

    def create_trans(self, arg, perm):
        return TensorHandle(np.transpose(arg.data, perm), arg.dtype.scalar)

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        a_data = a.data
        b_data = b.data
        if (a.dtype.primitive_bitwidth == 8 and a.dtype.is_floating()) or \
           (b.dtype.primitive_bitwidth == 8 and b.dtype.is_floating()):
            a_data = _convert_float(a_data, a.dtype, tl.float16, None).view(np.float16)
            b_data = _convert_float(b_data, b.dtype, tl.float16, None).view(np.float16)
        return TensorHandle(np.matmul(a_data, b_data, dtype=d.data.dtype) + d.data, d.dtype.scalar)

    def create_make_range(self, ret_ty, start, stop):
        return TensorHandle(np.arange(start, stop, dtype=np.int32), tl.int32)

    def create_histogram(self, data, bins, mask):
        if mask is None:
            mask = TensorHandle(np.ones_like(data.data, dtype=bool), tl.int1)

        # By default np.histogram returns int64 dtype values
        # Docs specify that returned dtype is taken based on optional weights.dtype
        # This is fix for interpreter cases where for example int32 tensor is being passed
        # But unexpectedly int64 values are being returned causing
        # tl.store to write 8 bytes instead of 4 bytes which lead to silent data corruption
        dummy_weights = np.ones_like(data.data, dtype=data.data.dtype)

        # force all masked elements to zero
        data = np.where(mask.data, data.data, np.zeros_like(data.data))
        histogram = np.histogram(data, bins=bins, range=(0, bins), weights=dummy_weights)[0]
        # remove overcounted elements
        histogram[0] -= np.logical_not(mask.data).sum()
        return TensorHandle(histogram, tl.int32)

    def create_gather(self, src, indices, axis):
        return TensorHandle(np.take_along_axis(src.data, indices.data, axis=axis), src.dtype.scalar)

    # pointer arithmetic

    def create_addptr(self, ptr, offset):
        dtype_tt = ptr.get_element_ty()
        element_bitwidth = dtype_tt.primitive_bitwidth
        # int1's bitwidth is 1, but we need to use 8 for pointer arithmetic
        element_bytewidth = max(1, element_bitwidth // 8)
        return TensorHandle(ptr.data + element_bytewidth * offset.data.astype(np.uint64), ptr.dtype)

    def create_expand_dims(self, arg, axis):
        return TensorHandle(np.expand_dims(arg.data, axis), arg.dtype.scalar)

    def create_broadcast(self, arg, shape):
        return TensorHandle(np.broadcast_to(arg.data, shape), arg.dtype.scalar)

    def create_cat(self, lhs, rhs):
        return TensorHandle(np.concatenate([lhs.data, rhs.data]), lhs.dtype.scalar)

    def create_join(self, lhs, rhs):
        # Triton only supports joining two original tensors into a new one along the last axis
        return TensorHandle(np.stack([lhs.data, rhs.data], axis=-1), lhs.dtype.scalar)

    def create_split(self, val):
        # Triton only supports splitting the original tensor into two along the last axis
        return (TensorHandle(val.data[..., 0], val.dtype.scalar), TensorHandle(val.data[..., 1], val.dtype.scalar))

    def create_splat(self, ret_ty, arg):
        shape = ret_ty.shape
        if isinstance(arg.dtype, tl.block_type):
            return TensorHandle(np.full(shape, arg.data[0], dtype=_get_np_dtype(arg.dtype)), arg.dtype.scalar)
        else:  # scalar
            return TensorHandle(np.full(shape, arg.data, dtype=_get_np_dtype(arg.dtype)), arg.dtype.scalar)

    def create_unsplat(self, arg):
        return TensorHandle(np.full((1, ), arg.data[0], dtype=_get_np_dtype(arg.dtype)), arg.dtype.scalar)

    def create_atomic_cas(self, ptr, cmp, val, sem, scope):
        if sem not in self.ir_sem_to_interpreter_sem:
            raise ValueError(f"unsupported semantic {sem}")
        sem = self.ir_sem_to_interpreter_sem[sem]
        return TensorHandle(_interpreter.atomic_cas(ptr.data, cmp.data, val.data, sem), cmp.dtype.scalar)

    def create_atomic_rmw(self, rmwOp, ptr, val, mask, sem, scope):
        if rmwOp not in self.ir_rmw_op_to_interpreter_rmw_op:
            raise ValueError(f"unsupported rmwOp {rmwOp}")
        if sem not in self.ir_sem_to_interpreter_sem:
            raise ValueError(f"unsupported semantic {sem}")
        rmwOp = self.ir_rmw_op_to_interpreter_rmw_op[rmwOp]
        sem = self.ir_sem_to_interpreter_sem[sem]
        return TensorHandle(_interpreter.atomic_rmw(rmwOp, ptr.data, val.data, mask.data, sem), val.dtype.scalar)

    def create_extern_elementwise(self, libName, libPath, symbol, argList, retType, isPure):
        raise NotImplementedError("extern_elementwise not supported in interpreter mode")

    def create_inline_asm(self, inlineAsm, constraints, values, type, isPure, pack):
        raise NotImplementedError("inline_asm not supported in interpreter mode")

    def create_print(self, prefix, hex, values, isSigned):
        # NOTE: the `isSigned` variable is not really used here; because Signness is already known
        # by `values` themselves in python interpreter, thus not really needed here;
        # it is only used for triton PrintOpToLLVM to correctly construct the format specifier.
        # Interpreter's device_print function has a different format than Triton's device_print
        msg = f"({self.grid_idx[0]}, {self.grid_idx[1]}, {self.grid_idx[2]})"
        if prefix:
            msg += f" {prefix}"
        if hex:
            np.set_printoptions(formatter={'all': lambda x: f"0x{x:02x}"})
        for value in values:
            print(msg + f" {value.data}")
        if hex:
            np.set_printoptions(formatter=None)

    def create_assert(self, condition, message):
        # Interpreter's device_assert function has a different format than Triton's device_assert
        assert condition, f"{message}"

    def create_assume(self, condition):
        assert condition, "Assume failed"

    def create_barrier(self):
        # Triton's barrier applies to each program in a grid, so it's a no-op in the interpreter
        pass

    def create_make_tensor_descriptor(self, base: TensorHandle, shape: List[TensorHandle], strides: List[TensorHandle],
                                      tensor_shape: List[int], is_signed: bool, padding: str = "zero"):
        desc = TensorDescHandle(base, shape, strides, tensor_shape, padding)
        desc.validate()
        return desc

    def create_descriptor_load(self, desc: TensorDescHandle, indices: List[TensorHandle], cache_modifier,
                               eviction_policy):
        assert isinstance(desc, TensorDescHandle)
        ptrs, mask = desc.materialize_pointers(indices)
        dtype_tt = ptrs.get_element_ty()
        dtype_np = _get_np_dtype(dtype_tt)
        padding = desc.padding
        if padding == _ir.PADDING_OPTION.PAD_ZERO:
            other = TensorHandle(np.zeros_like(ptrs.data, dtype=dtype_np), dtype_tt)
        elif padding == _ir.PADDING_OPTION.PAD_NAN:
            other = TensorHandle(np.full_like(ptrs.data, float('nan'), dtype=dtype_np), dtype_tt)
        else:
            raise ValueError(f"unsupported padding {padding}")
        return self.create_masked_load(ptrs, mask, other, cache_modifier=cache_modifier,
                                       eviction_policy=eviction_policy, is_volatile=False)

    def create_descriptor_store(self, desc: TensorDescHandle, value: TensorHandle, indices: List[TensorHandle]):
        ptrs, mask = desc.materialize_pointers(indices)
        return self.create_masked_store(ptrs, value, mask, None, None)

    def create_descriptor_gather(self, desc: TensorDescHandle, x_offsets: TensorHandle, y_offset: TensorHandle, type):
        dtype = desc.base.dtype.element_ty
        np_dtype = _get_np_dtype(dtype)
        result = np.zeros([x_offsets.data.shape[0], desc.block_shape[-1]], dtype=np_dtype)
        cache_modifier = None
        eviction_policy = None
        for i, x_offset in enumerate(x_offsets.data):
            indices = [TensorHandle(x_offset, tl.int32), y_offset]
            result[i, :] = self.create_descriptor_load(desc, indices, cache_modifier, eviction_policy).data
        return TensorHandle(result, dtype)

    def create_descriptor_scatter(self, desc: TensorDescHandle, value: TensorHandle, x_offsets: TensorHandle,
                                  y_offset: TensorHandle):
        for i, x_offset in enumerate(x_offsets.data):
            slice = TensorHandle(value.data[i], value.dtype)
            indices = [TensorHandle(x_offset, tl.int32), y_offset]
            self.create_descriptor_store(desc, slice, indices)

    def get_all_ones_value(self, type):
        np_type = _get_np_dtype(type)
        if "int" in np_type.name:
            return TensorHandle(np.full(1, -1, dtype=np_type), type.scalar)
        elif np_type == np.bool_:
            return TensorHandle(np.full(1, True, dtype=np_type), type.scalar)
        else:
            raise TypeError(f"unsupported type {type}")

    def create_dot_scaled(self, lhs: TensorHandle, lhs_scale_handle: Optional[TensorHandle],
                          lhs_format_enum: _ir.ScaleDotElemTypeTY, rhs: TensorHandle,
                          rhs_scale_handle: Optional[TensorHandle], rhs_format_enum: _ir.ScaleDotElemTypeTY,
                          fast_math: bool, lhs_k_pack: bool, rhs_k_pack: bool,
                          acc_handle: TensorHandle) -> TensorHandle:
        lhs_data = _prepare_dot_scaled_operand(lhs, lhs_scale_handle, lhs_format_enum, lhs_k_pack, is_rhs=False)
        rhs_data = _prepare_dot_scaled_operand(rhs, rhs_scale_handle, rhs_format_enum, rhs_k_pack, is_rhs=True)

        result = np.matmul(lhs_data, rhs_data) + acc_handle.data
        return TensorHandle(result, tl.float32)


_MISSING = object()
interpreter_builder = InterpreterBuilder()
interpreter_semantic: TritonSemantic = TritonSemantic(interpreter_builder)


class _LangPatchScope:
    """Tracks patched attributes so they can be restored."""

    def __init__(self) -> None:
        self._changes: list[tuple[object, str, object]] = []

    def set_attr(self, obj: object, name: str, value: object) -> None:
        original = getattr(obj, name, _MISSING)
        self._changes.append((obj, name, original))
        setattr(obj, name, value)

    def restore(self) -> None:
        while self._changes:
            obj, name, original = self._changes.pop()
            if original is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, original)


def _patch_attr(obj, name, member, builder, scope: _LangPatchScope):
    new_member = lambda *args, member=member, **kwargs: (member(*args, **
                                                                {k: v
                                                                 for k, v in kwargs.items()
                                                                 if k != "_semantic"}, _semantic=interpreter_semantic))
    scope.set_attr(obj, name, new_member)


def _patch_builtin(pkg, builder, scope: _LangPatchScope):
    for name, member in inspect.getmembers(pkg):
        if tl.core.is_builtin(member):
            _patch_attr(pkg, name, member, builder, scope)


def _patch_lang_tensor(tensor, scope: _LangPatchScope):

    def _get_bool(self):
        data = self.handle.data
        # in triton, only scalars can be converted to booleans
        # here we need this hack because all scalars are tensors
        return bool(data) if data.size == 1 else True

    def _get_transpose(self):
        handle = TensorHandle(np.transpose(self.handle.data), self.handle.dtype)
        assert self.type.is_block()
        block_shape = list(self.type.shape)
        block_shape[-1], block_shape[-2] = block_shape[-2], block_shape[-1]
        res_ty = tl.core.block_type(self.dtype, block_shape)
        return tl.core.tensor(handle, res_ty)

    scope.set_attr(tensor, "__index__", lambda self: int(self.handle.data.squeeze()))
    scope.set_attr(tensor, "__bool__", lambda self: _get_bool(self))
    scope.set_attr(tensor, "__repr__", lambda self: repr(self.handle.data))
    scope.set_attr(tensor, "__str__", lambda self: str(self.handle.data))
    scope.set_attr(tensor, "T", property(_get_transpose))


class ReduceScanOpInterface:

    def __init__(self, axis, combine_fn):
        self.axis = axis
        self.combine_fn = combine_fn

    def check_axis(self, shape, axis):
        if axis is not None and axis >= len(shape):
            raise ValueError(f"axis {axis} out of bounds for shape {shape}")

    def check_tensor(self, input):
        for arg in input:
            if not isinstance(arg, tl.core.tensor):
                raise ValueError(f"input must be a tensor, got {type(arg)}")
            self.check_axis(arg.shape, self.axis)

    def to_tensor(self, ret, dtype):
        np_dtype = _get_np_dtype(dtype)
        if hasattr(ret, "shape") and ret.shape:
            ret = ret.astype(np_dtype)
            ret_type = tl.block_type(dtype, list(ret.shape))
        else:
            ret = np.array([ret], dtype=np_dtype)
            ret_type = dtype
        return tl.core.tensor(TensorHandle(ret, dtype.scalar), ret_type)

    def apply_impl(self, input):
        raise NotImplementedError("apply_impl must be implemented by subclasses")

    def apply(self, input):
        if not isinstance(input, tuple):
            return self.apply((input, ))[0]
        self.check_tensor(input)
        ret = self.apply_impl(input)
        return tuple(ret) if isinstance(ret, (list, tuple)) else (ret, )


class ReduceOps(ReduceScanOpInterface):

    def __init__(self, axis, combine_fn, keep_dims):
        super().__init__(axis, combine_fn)
        self.keep_dims = keep_dims

    def unravel(self, input, axis):
        ret = []
        for data in input:
            if axis is not None:
                ret.append(data)
            else:
                axis = 0
                ret.append(self.to_tensor(data.handle.data.flatten(), data.dtype))
        return tuple(ret), axis

    def generic_reduce(self, input):
        original_axis = self.axis
        input, axis = self.unravel(input, self.axis)
        input_data = []
        output_data = []
        input_shape = input[0].handle.data.shape
        output_shape = input_shape[0:axis] + input_shape[axis + 1:]
        for arg in input:
            input_data.append(arg.handle.data)
            output_data.append(np.zeros(output_shape, dtype=arg.handle.data.dtype))
        # Reduce on axis
        for i in range(input_data[0].size):
            # Recover input_index from i using input_shape
            input_index = np.unravel_index(i, input_shape)
            output_index = input_index[0:axis] + input_index[axis + 1:]
            input_tuple = tuple(self.to_tensor(d[input_index], input[ii].dtype) for ii, d in enumerate(input_data))
            if input_index[axis] == 0:
                # First element
                for j in range(len(output_data)):
                    output_data[j][output_index] = input_tuple[j].handle.data.item()
            else:
                acc_tuple = tuple(self.to_tensor(o[output_index], input[oi].dtype) for oi, o in enumerate(output_data))
                combine_fn_ret = self.combine_fn.fn(*acc_tuple, *input_tuple)
                acc_tuple = (combine_fn_ret, ) if not isinstance(combine_fn_ret, tuple) else combine_fn_ret
                for j in range(len(output_data)):
                    output_data[j][output_index] = acc_tuple[j].handle.data.item() if isinstance(
                        acc_tuple[j], tl.core.tensor) else acc_tuple[j]
        # Pack output
        ret = []
        for i, data in enumerate(output_data):
            if self.keep_dims:
                if original_axis is not None:
                    data = np.expand_dims(data, axis)
                else:
                    for _ in range(len(input_shape)):
                        data = np.expand_dims(data, 0)

            elif original_axis is None:
                # Take a scalar
                data = data.item()
            ret.append(self.to_tensor(data, input[i].dtype))
        return ret

    def min_max(self, input, val_reduce_op, idx_reduce_op=None):
        # If input is a tuple, it must be (val, index), and we only take val
        input = input[0] if isinstance(input, tuple) else input
        val = None
        idx = None
        if val_reduce_op:
            val = self.to_tensor(val_reduce_op(input.handle.data, axis=self.axis, keepdims=self.keep_dims), input.dtype)
        if idx_reduce_op:
            idx = self.to_tensor(idx_reduce_op(input.handle.data, axis=self.axis, keepdims=self.keep_dims), tl.int32)
        if val is not None and idx is not None:
            return val, idx
        elif val is not None:
            return val
        elif idx is not None:
            return idx
        else:
            raise ValueError("val_reduce_op and idx_reduce_op are both None")

    def sum(self, input):
        return self.to_tensor(np.sum(input.handle.data, axis=self.axis, keepdims=self.keep_dims), input.dtype)

    def apply_impl(self, input):
        # Note: np.nanargmin/np.nanargmax always return the leftmost index
        # for equal values, whereas tie_break_fast on hardware returns an
        # arbitrary index. This is a known remaining divergence between the
        # interpreter and JIT for inputs with equal non-NaN elements.
        if (self.combine_fn is tl.standard._argmin_combine_tie_break_left
                or self.combine_fn is tl.standard._argmin_combine_tie_break_fast):
            return self.min_max(input[0], val_reduce_op=np.nanmin, idx_reduce_op=np.nanargmin)
        elif (self.combine_fn is tl.standard._argmax_combine_tie_break_left
              or self.combine_fn is tl.standard._argmax_combine_tie_break_fast):
            return self.min_max(input[0], val_reduce_op=np.nanmax, idx_reduce_op=np.nanargmax)
        elif self.combine_fn is tl.standard._elementwise_max:
            return self.min_max(input[0], val_reduce_op=np.nanmax, idx_reduce_op=None)
        elif self.combine_fn is tl.standard._elementwise_min:
            return self.min_max(input[0], val_reduce_op=np.nanmin, idx_reduce_op=None)
        elif self.combine_fn is tl.standard._sum_combine:
            return self.sum(input[0])
        else:
            # Fall back to the slow mode
            return self.generic_reduce(input)


class ScanOps(ReduceScanOpInterface):

    def __init__(self, axis, combine_fn, reverse):
        super().__init__(axis, combine_fn)
        self.reverse = reverse

    def cumsum(self, input):
        return [self.to_tensor(np.cumsum(input.handle.data, axis=self.axis), dtype=input.dtype)]

    def cumprod(self, input):
        return [self.to_tensor(np.cumprod(input.handle.data, axis=self.axis), dtype=input.dtype)]

    def generic_scan(self, input):
        input_data = []
        output_data = []
        shape = input[0].handle.data.shape
        for arg in input:
            input_data.append(arg.handle.data)
            output_data.append(np.zeros(shape, dtype=arg.handle.data.dtype))
        # Scan on axis
        for i in range(input_data[0].size):
            # Recover index from i using shape
            index = np.unravel_index(i, shape)
            data = tuple(self.to_tensor(d[index], input[ii].dtype) for ii, d in enumerate(input_data))
            if index[self.axis] == 0:
                # First element
                for j in range(len(output_data)):
                    output_data[j][index] = data[j].handle.data.item()
            else:
                prev_index = tuple(index[i] - 1 if i == self.axis else index[i] for i in range(len(index)))
                acc_tuple = tuple(self.to_tensor(o[prev_index], input[oi].dtype) for oi, o in enumerate(output_data))
                combine_fn_ret = self.combine_fn.fn(*acc_tuple, *data)
                acc_tuple = (combine_fn_ret, ) if not isinstance(combine_fn_ret, tuple) else combine_fn_ret
                for j in range(len(output_data)):
                    output_data[j][index] = acc_tuple[j].handle.data.item() if isinstance(
                        acc_tuple[j], tl.core.tensor) else acc_tuple[j]
        # Pack output
        ret = []
        for i, data in enumerate(output_data):
            ret.append(self.to_tensor(data, input[i].dtype))
        return ret

    def apply_impl(self, input):
        new_input = []
        if self.reverse:
            for arg in input:
                new_input.append(self.to_tensor(np.flip(arg.handle.data, axis=self.axis), arg.dtype))
        else:
            new_input = input
        if self.combine_fn == tl.standard._sum_combine:
            ret = self.cumsum(new_input[0])
        elif self.combine_fn == tl.standard._prod_combine:
            ret = self.cumprod(new_input[0])
        else:
            # Fall back to the slow mode
            ret = self.generic_scan(new_input)
        if self.reverse:
            for arg in ret:
                arg.handle.data = np.flip(arg.handle.data, axis=self.axis)
        return ret


def _patch_reduce_scan(scope: _LangPatchScope):
    # Because interpreter doesn't support region_builder_fn, we cannot patch the builder
    # to use the new reduce and scan functions.
    # Instead, we need to patch reduce and reduce functions in tl and tl.core
    def _new_reduce(input, axis, combine_fn, keep_dims=False, **kwargs):
        return ReduceOps(axis, combine_fn, keep_dims).apply(input)

    def _new_scan(input, axis, combine_fn, reverse=False, **kwargs):
        return ScanOps(axis, combine_fn, reverse).apply(input)

    scope.set_attr(tl, "reduce", _new_reduce)
    scope.set_attr(tl, "associative_scan", _new_scan)
    scope.set_attr(tl.core, "reduce", _new_reduce)
    scope.set_attr(tl.core, "associative_scan", _new_scan)


def _patch_lang_core(lang, scope: _LangPatchScope):

    def _new_to_ir(self, builder):
        # We need to specify signedness for integer types in the numpy mode
        if self.name == 'void':
            return builder.get_void_ty()
        elif self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name == 'int8':
            return builder.get_int8_ty()
        elif self.name == 'uint8':
            return builder.get_uint8_ty()
        elif self.name == 'int16':
            return builder.get_int16_ty()
        elif self.name == 'uint16':
            return builder.get_uint16_ty()
        elif self.name == 'int32':
            return builder.get_int32_ty()
        elif self.name == 'uint32':
            return builder.get_uint32_ty()
        elif self.name == 'int64':
            return builder.get_int64_ty()
        elif self.name == 'uint64':
            return builder.get_uint64_ty()
        elif self.name == 'fp8e5':
            return builder.get_fp8e5_ty()
        elif self.name == 'fp8e4nv':
            return builder.get_fp8e4nv_ty()
        elif self.name == 'fp8e4b15':
            return builder.get_fp8e4b15_ty()
        elif self.name == 'fp8e5b16':
            return builder.get_fp8e5b16_ty()
        elif self.name == 'fp8e4b8':
            return builder.get_fp8e4b8_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'bf16':
            return builder.get_bf16_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        elif self.name == 'fp64':
            return builder.get_double_ty()
        raise ValueError(f'fail to convert {self} to ir type')

    # can't just map lang.static_range to `range`, because `tl.static_range`
    # can get `step` passed by keyword
    def _new_range(arg1, arg2=None, step=None, **kwargs):
        if step is None:
            step = 1
        if arg2 is None:
            start, end = 0, arg1
        else:
            start, end = arg1, arg2
        return range(start, end, step)

    def _new_static_assert(cond, msg=""):
        assert cond, msg

    def _set_attr(input, values, name):
        # skip non tensor types. This may happen for induction variables.
        if not isinstance(input, tl.tensor):
            return input
        # Unwrap constexpr
        values = [values] if not isinstance(values, (list, tuple)) else values
        values = [v.value if isinstance(v, tl.constexpr) else v for v in values]
        if len(values) != max(1, len(input.shape)):
            raise ValueError(f"len(values) != len(input.shape) for {name}")
        input.handle.set_attr(name, values)
        return input

    scope.set_attr(lang, "range", _new_range)
    scope.set_attr(lang, "static_range", _new_range)
    scope.set_attr(lang, "static_assert", _new_static_assert)
    scope.set_attr(lang, "static_print", print)
    scope.set_attr(lang.dtype, "to_ir", _new_to_ir)
    scope.set_attr(lang, "multiple_of", partial(_set_attr, name="tt.divisibility"))
    scope.set_attr(lang, "max_contiguous", partial(_set_attr, name="tt.contiguity"))
    scope.set_attr(lang, "max_constancy", partial(_set_attr, name="tt.constancy"))

    _patch_reduce_scan(scope)


def _patch_lang(fn):
    scope = _LangPatchScope()
    langs = [value for _, value in fn.__globals__.items() if inspect.ismodule(value) and value in [tl, tl.core]]
    assert len(langs) >= 1, "triton.language must be visible from within jit'd function"
    for lang in langs:
        _patch_builtin(lang, interpreter_builder, scope)
        _patch_builtin(lang.tensor, interpreter_builder, scope)
        if lang == tl:
            _patch_builtin(lang.math, interpreter_builder, scope)
        _patch_lang_tensor(lang.tensor, scope)
        _patch_lang_core(lang, scope)
    _patch_builtin(tl.core.tensor_descriptor_base, interpreter_builder, scope)
    return scope


# TODO: wrap everything in triton tensors
def _implicit_cvt(arg):
    if isinstance(arg, bool):
        handle = TensorHandle(np.array([arg], dtype=np.bool_), tl.int1)
        return tl.tensor(handle, tl.int1)
    if isinstance(arg, int):
        ty = tl.str_to_ty(triton.runtime.jit.mangle_type(arg), None)
        dtype = np.int32
        if -2**31 <= arg < 2**31:
            dtype = np.int32
        elif 2**31 <= arg < 2**32:
            dtype = np.uint32
        elif -2**63 <= arg < 2**63:
            dtype = np.int64
        elif 2**63 <= arg < 2**64:
            dtype = np.uint64
        else:
            raise ValueError(f"Unsupported integer value {arg}")
        handle = TensorHandle(np.array([arg], dtype=dtype), ty)
        return tl.tensor(handle, ty)
    if hasattr(arg, "data_ptr"):
        ty = tl.str_to_ty(triton.runtime.jit.mangle_type(arg), None)
        handle = TensorHandle(np.array([arg.data_ptr()], dtype=np.uint64), ty)
        return tl.tensor(handle, ty)
    elif isinstance(arg, tuple):
        return _tuple_create(arg, map(_implicit_cvt, arg))
    elif isinstance(arg, TensorDescriptor):
        strides = [_implicit_cvt(s) for s in arg.strides]
        assert arg.strides[-1] == 1
        strides[-1] = tl.constexpr(1)
        return interpreter_semantic.make_tensor_descriptor(base=_implicit_cvt(arg.base),
                                                           shape=[_implicit_cvt(s) for s in arg.shape], strides=strides,
                                                           block_shape=[tl.constexpr(b) for b in arg.block_shape],
                                                           padding_option=arg.padding)
    return arg


def _unwrap_tensor(t):
    if isinstance(t, triton.runtime.jit.TensorWrapper):
        return t.base
    return t


def _rewrap_tensor(t, original_tensor):
    if isinstance(original_tensor, triton.runtime.jit.TensorWrapper):
        return triton.runtime.jit.TensorWrapper(t, original_tensor.dtype)
    return t


class GridExecutor:

    def __init__(self, fn, arg_names, grid, pre_run_hooks=[]):
        from .jit import _normalize_ty  # TODO: modularize

        self.fn = fn
        self.arg_names = arg_names
        self.grid = grid
        self.pre_run_hooks = pre_run_hooks
        __annotations__ = {name: _normalize_ty(ty) for name, ty in inspect.get_annotations(fn).items()}
        self.constexprs = [name for name in arg_names if __annotations__.get(name) == "constexpr"]

    def _init_args_hst(self, args_dev, kwargs):
        storages = {}

        def _to_cpu(arg):
            if isinstance(arg, tuple):
                return _tuple_create(arg, map(_to_cpu, arg))
            elif isinstance(arg, TensorDescriptor):
                return TensorDescriptor(
                    _to_cpu(arg.base),
                    arg.shape,
                    arg.strides,
                    arg.block_shape,
                    arg.padding,
                    arg.round_f32_to_tf32,
                )
            elif not hasattr(arg, "data_ptr"):
                return arg

            unwrapped_arg = _unwrap_tensor(arg)
            if unwrapped_arg.untyped_storage().data_ptr() not in storages:
                storage = unwrapped_arg.untyped_storage()
                storages[storage.data_ptr()] = storage.cpu()

            storage = storages[unwrapped_arg.untyped_storage().data_ptr()]
            cpu_arg = unwrapped_arg.new_empty(0, device='cpu')
            cpu_arg.set_(storage, unwrapped_arg.storage_offset(), unwrapped_arg.size(), unwrapped_arg.stride())
            cpu_arg = _rewrap_tensor(cpu_arg, original_tensor=arg)
            return cpu_arg

        args_hst = [_to_cpu(arg) for arg in args_dev]

        # Process keyword arguments
        kwargs_hst = {}
        for key, value in kwargs.items():
            kwargs_hst[key] = _to_cpu(value)
        return args_hst, kwargs_hst

    def _restore_args_dev(self, args_dev, args_hst, kwargs, kwargs_hst):
        storages = {}

        def _from_cpu(arg_dev, arg_hst):
            if hasattr(arg_dev, "data_ptr"):
                # No need to rewrap because this just modifies internal
                arg_dev, arg_hst = _unwrap_tensor(arg_dev), _unwrap_tensor(arg_hst)
                storages[arg_dev.untyped_storage().data_ptr()] = (arg_dev.untyped_storage(), arg_hst.untyped_storage())
            elif isinstance(arg_dev, tuple):
                for (arg_dev, arg_hst) in zip(arg_dev, arg_hst):
                    _from_cpu(arg_dev, arg_hst)
            elif isinstance(arg_dev, TensorDescriptor):
                _from_cpu(arg_dev.base, arg_hst.base)

        for arg_dev, arg_hst in zip(args_dev, args_hst):
            _from_cpu(arg_dev, arg_hst)

        # Restore keyword arguments
        for key, kwarg_dev in kwargs.items():
            kwarg_hst = kwargs_hst[key]
            _from_cpu(kwarg_dev, kwarg_hst)

        for (arg_dev, arg_hst) in storages.values():
            arg_dev.copy_(arg_hst)

    def __call__(self, *args_dev, **kwargs):
        # Removes not used reserved keywords from kwargs
        # Triton doesn't support keyword-only, variable positional or variable keyword arguments
        # It's safe to inspect only positional or keyword arguments (i.e., argspec.args)
        argspec = inspect.getfullargspec(self.fn)
        kwargs = {k: v for k, v in kwargs.items() if k in argspec.args}
        # copy arguments to the host
        args_hst, kwargs_hst = self._init_args_hst(args_dev, kwargs)
        # run pre-run hooks
        for hook in self.pre_run_hooks:
            hook(*args_hst, **kwargs_hst)
        # remaps core language functions to interpreted ones
        patch_scope = _patch_lang(self.fn)
        try:
            # we need to copy arguments to the host for the interpreter
            # implicitly convert tensor arguments to their base pointers
            args = inspect.getcallargs(self.fn, *args_hst, **kwargs_hst)
            args = {name: arg if name in self.constexprs else _implicit_cvt(arg) for name, arg in args.items()}
            # iterate through grid
            grid = self.grid(args) if callable(self.grid) else self.grid
            assert len(grid) <= 3, "grid must have at most 3 dimensions"
            grid = grid + (1, ) * (3 - len(grid))
            interpreter_builder.set_grid_dim(*grid)
            try:
                for x in range(grid[0]):
                    for y in range(grid[1]):
                        for z in range(grid[2]):
                            interpreter_builder.set_grid_idx(x, y, z)
                            self.fn(**args)
            except Exception as e:
                if triton.knobs.compilation.front_end_debugging:
                    raise
                raise InterpreterError(repr(e)) from e
        finally:
            patch_scope.restore()
        # copy arguments back to propagate side-effects
        self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)


class ASTTransformer(ast.NodeTransformer):

    def visit_Assign(self, node):
        names = []
        for target in node.targets:
            names += [self.visit(target)]
        if len(names) > 1:
            raise ValueError("Multiple assignments are not supported")
        # Modify the assignment x = value to
        # interpreter_semantic.to_tensor(value, False)
        node.value = ast.Call(
            func=ast.Attribute(value=ast.Name(id="interpreter_semantic", ctx=ast.Load()), attr="to_tensor",
                               ctx=ast.Load()), args=[node.value, ast.Constant(value=False)], keywords=[])
        return node


class FunctionRewriter:
    ast_transformer = ASTTransformer()

    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.kwargs = kwargs
        self.filename: str = ""
        # Absolute line number in the file
        self.def_file_lineno: int = 0

    def rewrite_ast(self):
        # If exception is raise, it means the function does not have source code available,
        # e.g., dynamically generated functions, we cannot rewrite it so just return the original function
        try:
            lines, _ = inspect.getsourcelines(self.fn)
        except Exception:
            return self.fn

        # truncate lines before def
        # @triton.autotune(...)
        # ...
        # @triton.jit
        # ...
        # def foo(...): <- this line is the function definition
        self.filename, self.def_file_lineno = self._get_jit_fn_file_line()
        self.def_lineno = self._find_def(lines)
        src = self._prepare_source(lines)
        transformed_ast = self._transform_ast(src)
        return self._compile_and_exec(transformed_ast)

    def _get_jit_fn_file_line(self):
        from .jit import JITFunction
        jit_func = JITFunction(self.fn)
        return jit_func.file_name, jit_func.def_file_line_number

    def _find_def(self, lines):
        def_lineno = 0
        # Line numbers start from 1
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                def_lineno = i + 1
        return def_lineno

    def _prepare_source(self, lines):
        lines = lines[self.def_lineno - 1:]
        src = ''.join(lines)
        return textwrap.dedent(src)

    def _transform_ast(self, src):
        # src is like:
        # 1: def foo(...):
        # 2:  ...
        parsed_ast = ast.parse(src)
        transformed_ast = self.ast_transformer.visit(parsed_ast)
        ast.fix_missing_locations(transformed_ast)
        inc_lineno = self.def_file_lineno - 1
        ast.increment_lineno(transformed_ast, inc_lineno)
        return transformed_ast

    def _compile_and_exec(self, transformed_ast):
        compiled_code = compile(transformed_ast, filename=self.filename, mode='exec')
        local_namespace = {**self.kwargs}
        fn_globals = self.fn.__globals__
        for key, value in globals().items():
            if key not in fn_globals:
                fn_globals[key] = value
        exec(compiled_code, fn_globals, local_namespace)
        return local_namespace[self.fn.__name__]


class InterpretedFunction(KernelInterface[T]):
    # Cache all rewritten functions
    rewritten_fn: Dict[Callable, Callable] = {}

    def __init__(self, fn, **kwargs) -> None:
        self.fn = fn
        self.rewriter = FunctionRewriter(fn, **kwargs)
        self.kwargs = kwargs
        self.pre_run_hooks = []

        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]

    def run(self, *args, grid, warmup, **kwargs):
        if warmup:
            return
        fn = self.rewrite()
        return GridExecutor(fn, self.arg_names, grid, self.pre_run_hooks)(*args, **kwargs)

    def add_pre_run_hook(self, hook):
        assert callable(hook)
        self.pre_run_hooks.append(hook)

    def rewrite(self):
        if self.fn not in self.rewritten_fn:
            self.rewritten_fn[self.fn] = self.rewriter.rewrite_ast()
        return self.rewritten_fn[self.fn]

    @property
    def __name__(self):
        return self.fn.__name__

    def __call__(self, *args, **kwargs):
        # This is a device function call
        _patch_lang(self.fn)
        fn = self.rewrite()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise InterpreterError(repr(e)) from e
<|endoftext|>from typing import Optional, Protocol
from contextvars import ContextVar


class Buffer(Protocol):

    def data_ptr(self) -> int:
        ...


class Allocator(Protocol):

    def __call__(self, size: int, alignment: int, stream: Optional[int]) -> Buffer:
        ...


class NullAllocator:

    def __call__(self, size: int, alignment: int, stream: Optional[int]) -> Buffer:
        raise RuntimeError("Kernel requires a runtime memory allocation, but no allocator was set. " +
                           "Use triton.set_allocator to specify an allocator.")


_NULL_ALLOCATOR = NullAllocator()

_allocator: ContextVar[Allocator] = ContextVar("_allocator", default=_NULL_ALLOCATOR)


def set_allocator(allocator: Allocator) -> None:
    """
    The allocator function is called during kernel launch for kernels that
    require additional global memory workspace.
    """
    _allocator.set(allocator)


class _AllocatorWrapper:
    """
    Wrapper to provide ContextVar-like .get()/.set() methods. profile_allocator is
    used in same way as allocator so it is useful to maintain the interface.
    """

    def __init__(self, allocator: Allocator) -> None:
        self._allocator = allocator

    def get(self) -> Allocator:
        return self._allocator

    def set(self, allocator: Allocator) -> None:
        self._allocator = allocator

    def __call__(self, size: int, alignment: int, stream: Optional[int]) -> Buffer:
        return self._allocator(size, alignment, stream)


_profile_allocator = _AllocatorWrapper(_NULL_ALLOCATOR)


def set_profile_allocator(allocator: Optional[Allocator]) -> None:
    """
    The profile allocator function is called before kernel launch for kernels
    that require additional global memory workspace.
    """
    _profile_allocator.set(allocator if allocator is not None else _NULL_ALLOCATOR)


def has_profile_allocator() -> bool:
    return not isinstance(_profile_allocator.get(), NullAllocator)
<|endoftext|>from __future__ import annotations

import builtins
import time
import inspect
import hashlib
import json
from functools import cached_property
from typing import Dict, Tuple, List, Optional

from .. import knobs
from .jit import KernelInterface, JITFunction
from .errors import OutOfResources, PTXASError, AutotunerError
from .driver import driver
from .cache import get_cache_manager, triton_key
from triton._C.libtriton import get_cache_invalidating_env_vars


class Autotuner(KernelInterface):

    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=None, post_hook=None,
                 prune_configs_by: Optional[Dict] = None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None,
                 cache_results=False):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'early_config_prune': a function used to prune configs. It should have the signature
                `prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
                and return pruned configs. It should return at least one config.
        """
        if not configs:
            self.configs = [Config({}, num_warps=4, num_stages=3, num_ctas=1)]
        else:
            self.configs = configs
        self.keys = key
        self.cache: Dict[Tuple, Config] = {}
        self.arg_names = arg_names
        self.cache_results = (cache_results or knobs.autotuning.cache) and not knobs.runtime.interpret

        # Reset to zero or restore values
        self.reset_to_zero = []
        if reset_to_zero is not None:
            self.reset_to_zero = list(reset_to_zero)
        self.restore_value = []
        if restore_value is not None:
            self.restore_value = list(restore_value)

        # Hook to reset or restore for required tensors
        self.pre_hook = lambda kwargs, reset_only=False: 0
        self.post_hook = lambda kwargs, exception: 0
        self.user_defined_pre_hook = False
        self.user_defined_post_hook = False
        if pre_hook:
            self.pre_hook = pre_hook
            self.user_defined_pre_hook = True
        elif (len(self.reset_to_zero) > 0 or len(self.restore_value) > 0):

            def _pre_hook(kwargs, reset_only=False):
                for name in self.reset_to_zero:
                    if kwargs[name] is not None:
                        kwargs[name].zero_()
                if not reset_only:
                    self.restore_copies = {
                        name: kwargs[name].clone()
                        for name in self.restore_value
                        if kwargs[name] is not None
                    }

            self.pre_hook = _pre_hook

        if post_hook:
            self.post_hook = post_hook
            self.user_defined_post_hook = True
        elif len(self.restore_value) > 0:

            def _post_hook(kwargs, exception):
                for name, value in self.restore_copies.items():
                    kwargs[name].copy_(value)
                self.restore_copies = {}

            self.post_hook = _post_hook

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get("early_config_prune", self.early_config_prune)

        self.fn = fn
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn

        self._do_bench = do_bench
        self.num_warmups = warmup
        self.num_reps = rep
        self.use_cuda_graph = use_cuda_graph

        # If we got explicitly called via the old interface, raise a warning
        # and proceed with the old behavior.
        if warmup is not None or rep is not None or use_cuda_graph:
            import warnings
            warnings.warn(("warmup, rep, and use_cuda_graph parameters are deprecated. See "
                           "https://github.com/triton-lang/triton/pull/4496 for details."), DeprecationWarning,
                          stacklevel=1)
            if use_cuda_graph:
                from ..testing import do_bench_cudagraph
                self._do_bench = lambda kernel_call, quantiles: do_bench_cudagraph(
                    kernel_call,
                    rep=rep if rep is not None else 100,
                    quantiles=quantiles,
                )
                return

            import triton.testing
            self._do_bench = lambda kernel_call, quantiles: triton.testing.do_bench(
                kernel_call,
                warmup=warmup if warmup is not None else 25,
                rep=rep if rep is not None else 100,
                quantiles=quantiles,
            )
            return

    @cached_property
    def do_bench(self):
        if self._do_bench is None:
            return driver.active.get_benchmarker()
        return self._do_bench

    def _bench(self, *args, config, **meta):
        from ..compiler.errors import CompileTimeAssertionFailure

        verbose = knobs.autotuning.print
        if verbose:
            print(f"Autotuning kernel {self.base_fn.__name__} with config {config}")

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                             " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(full_nargs)
            try:
                self.fn.run(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    self.post_hook(full_nargs, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            self.post_hook(full_nargs, exception=None)

        try:
            return self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
        except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as e:
            if verbose:
                print(f"Autotuning failed with {e}")
            return [float("inf"), float("inf"), float("inf")]

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        # We can't serialize prehooks, so just give up and run the benchmarks.
        if not tuning_key or any(cfg.pre_hook for cfg in configs):
            bench_fn()
            return False

        from triton.compiler.compiler import make_backend

        fn = self.fn
        while not isinstance(fn, JITFunction):
            fn = fn.fn

        env_vars = get_cache_invalidating_env_vars()
        cache_key = [
            triton_key(),
            make_backend(driver.active.get_current_target()).hash(),
            fn.cache_key,
            str(sorted(env_vars.items())),
            str(tuning_key),
        ] + [str(c) for c in configs]
        cache_key = hashlib.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
        cache = get_cache_manager(cache_key)
        file_name = f"{fn.__name__[:150]}.autotune.json"
        path = cache.get_file(file_name)
        if path:
            with open(path, "r") as cached_configs:
                timings = json.load(cached_configs)["configs_timings"]
                timings = {Config(**config): timing for config, timing in timings}
                self.cache[tuning_key] = builtins.min(timings, key=timings.get)
                self.configs_timings = timings
            return True

        bench_fn()
        cache.put(
            json.dumps({
                "key":
                tuning_key,
                "configs_timings":
                [(config.__dict__, timings) for config, timings in self.configs_timings.items() if not config.pre_hook],
            }), file_name, binary=False)
        return False

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            key = [_args[key] for key in self.keys if key in _args]
            for _, arg in _args.items():
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)

                if len(pruned_configs) == 1:
                    # Match single-config autotune behavior: no benchmarking is needed.
                    self.cache[key] = pruned_configs[0]
                    used_cached_result = True
                else:

                    def benchmark():
                        bench_start = time.time()
                        timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                        bench_end = time.time()
                        self.bench_time = bench_end - bench_start
                        self.cache[key] = builtins.min(timings, key=timings.get)
                        full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                        self.pre_hook(full_nargs, reset_only=True)
                        self.configs_timings = timings

                    if self.cache_results:
                        used_cached_result = self.check_disk_cache(key, pruned_configs, benchmark)
                    else:
                        benchmark()

                    if knobs.autotuning.listener is not None:
                        jit_fn = self.fn
                        while not isinstance(jit_fn, JITFunction):
                            jit_fn = jit_fn.fn
                        knobs.autotuning.listener(
                            fn=jit_fn,
                            key=key,
                            best_config=self.cache[key],
                            configs_timings=self.configs_timings,
                            duration=getattr(self, 'bench_time', None) if not used_cached_result else None,
                            cache_hit=used_cached_result,
                        )

            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if knobs.autotuning.print and not used_cached_result:
            print(f"Triton autotuning for function {self.base_fn.__name__},\nwith key as {key},\n"
                  f"finished after {self.bench_time:.2f}s,\nbest config selected: {self.best_config};")
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret

    def prune_configs(self, kwargs: Dict) -> List[Config]:
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
            if not pruned_configs:
                raise AutotunerError(
                    "No valid autotuner configs after pruning. `early_config_prune` should return at least one config.")
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            elif not isinstance(top_k, int):
                # Slice index must be an integer
                raise TypeError("Error while pruning configs, top_k must be either 1) a float <= 1.0 or 2) an int")

            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for autotune_config in self.prune_configs(kwargs):
            ret.append(self.fn.warmup(
                *args,
                **kwargs,
                **autotune_config.all_kwargs(),
            ))
        self.nargs = None
        return ret


class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_stages: int
    :ivar num_ctas: number of blocks in a block cluster. SM90+ only.
    :type num_ctas: int
    :type maxnreg: Optional[int]
    :ivar maxnreg: maximum number of registers one thread can use.  Corresponds
                       to ptx .maxnreg directive.  Not supported on all platforms.
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    :ivar ir_override: filename of a user-defined IR (*.{ttgir|llir|ptx|amdgcn}).
    """

    def __init__(self, kwargs, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.num_stages = num_stages
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook
        self.ir_override = ir_override

    def __setstate__(self, state):
        self.kwargs = state.get("kwargs", {})
        self.num_warps = state.get("num_warps", 4)
        self.num_stages = state.get("num_stages", 3)
        self.num_ctas = state.get("num_ctas", 1)
        self.maxnreg = state.get("maxnreg", None)
        self.pre_hook = state.get("pre_hook", None)
        self.ir_override = state.get("ir_override", None)

    def all_kwargs(self):
        return {
            **self.kwargs, **{
                k: v
                for (k, v) in (
                    ("num_warps", self.num_warps),
                    ("num_ctas", self.num_ctas),
                    ("num_stages", self.num_stages),
                    ("maxnreg", self.maxnreg),
                    ("ir_override", self.ir_override),
                ) if v is not None
            }
        }

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        res.append(f"num_warps: {self.num_warps}")
        res.append(f"num_ctas: {self.num_ctas}")
        res.append(f"num_stages: {self.num_stages}")
        res.append(f"maxnreg: {self.maxnreg}")
        return ", ".join(res)

    def __hash__(self):
        return hash((*self.all_kwargs().items(), self.pre_hook))

    def __eq__(self, other):
        self_tuple = tuple((
            *self.all_kwargs().items(),
            self.pre_hook,
        ))
        other_tuple = tuple((
            *other.all_kwargs().items(),
            other.pre_hook,
        ))
        return self_tuple == other_tuple


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune': a function used to prune configs. It should have the signature
                `prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
                and return pruned configs. It should return at least one config.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: warmup time (in ms) to pass to benchmarking (deprecated).
    :type warmup: int
    :param rep: repetition time (in ms) to pass to benchmarking (deprecated).
    :type rep: int
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    :param cache_results: whether to cache autotune timings to disk.  Defaults to False.
    "type cache_results: bool
    """

    def decorator(fn):
        return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, do_bench=do_bench, cache_results=cache_results)

    return decorator


class Heuristics(KernelInterface):

    def __init__(self, fn, arg_names, values) -> None:
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)


def heuristics(values):
    """
    Decorator for specifying how the values of certain meta-parameters may be computed.
    This is useful for cases where auto-tuning is prohibitively expensive, or just not applicable.

    .. highlight:: python
    .. code-block:: python

        # smallest power-of-two >= x_size
        @triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
        @triton.jit
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...
    :param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
                   each such function takes a dict of all the arguments passed to the kernel, keyed by argument name, as input.
    :type values: dict[str, Callable[[dict[str, Any]], Any]]
    """

    def decorator(fn):
        return Heuristics(fn, fn.arg_names, values)

    return decorator
<|endoftext|>from __future__ import annotations
from typing import Callable, Optional
from concurrent.futures import Executor, as_completed, Future
from contextvars import ContextVar

active_mode: ContextVar[Optional[AsyncCompileMode]] = ContextVar("async_compile_active_mode", default=None)


class FutureKernel:

    def __init__(self, finalize_compile: Callable, future: Future):
        self.finalize_compile = finalize_compile
        self.kernel = None
        self.future = future

    def result(self, ignore_errors: bool = False):
        if self.kernel is not None:
            return self.kernel

        try:
            kernel = self.future.result()
        except Exception:
            if ignore_errors:
                return
            else:
                raise
        self.finalize_compile(kernel)
        self.kernel = kernel
        return kernel

    def __getattr__(self, name):
        # Defer to the compiled kernel so users can interact with this object
        # like a normal CompiledKernel without needing to call result() first.
        return getattr(self.result(), name)


class AsyncCompileMode:

    def __init__(self, executor: Executor, *, ignore_errors=False):
        self.executor = executor
        self.ignore_errors = ignore_errors
        self.raw_futures = []
        self.future_kernels = {}

    def submit(self, key, compile_fn, finalize_fn):
        future = self.future_kernels.get(key)
        if future is not None:
            return future

        future = self.executor.submit(compile_fn)
        future._key = key
        self.raw_futures.append(future)
        future_kernel = FutureKernel(finalize_fn, future)
        self.future_kernels[key] = future_kernel
        return future_kernel

    def __enter__(self):
        if active_mode.get() is not None:
            raise RuntimeError("Another AsyncCompileMode is already active")
        active_mode.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        active_mode.set(None)
        # Finalize any outstanding compiles
        for future in as_completed(self.raw_futures):
            self.future_kernels[future._key].result(self.ignore_errors)
<|endoftext|>from .autotuner import (Autotuner, Config, Heuristics, autotune, heuristics)
from .cache import RedisRemoteCacheBackend, RemoteCacheBackend
from .driver import driver
from .jit import JITFunction, KernelInterface, MockTensor, TensorWrapper, reinterpret
from .errors import OutOfResources, InterpreterError

__all__ = [
    "autotune",
    "Autotuner",
    "Config",
    "driver",
    "Heuristics",
    "heuristics",
    "InterpreterError",
    "JITFunction",
    "KernelInterface",
    "MockTensor",
    "OutOfResources",
    "RedisRemoteCacheBackend",
    "reinterpret",
    "RemoteCacheBackend",
    "TensorWrapper",
]
<|endoftext|>from __future__ import annotations

import os

from ..backends import backends, DriverBase


def _create_driver() -> DriverBase:
    selected = os.environ.get("TRITON_DEFAULT_BACKEND", None)
    if selected:
        if selected not in backends:
            raise RuntimeError(f"Unknown backend device '{selected}'. Available backends: {list(backends.keys())}")
        driver = backends[selected].driver
        if not driver.is_active():
            raise RuntimeError(f"Backend device '{selected}' is not active.")
        return driver()
    else:
        active_drivers = [x.driver for x in backends.values() if x.driver.is_active()]
        if len(active_drivers) != 1:
            raise RuntimeError(f"{len(active_drivers)} active drivers ({active_drivers}). There should only be one.")
        return active_drivers[0]()


class DriverConfig:

    def __init__(self) -> None:
        self._default: DriverBase | None = None
        self._active: DriverBase | None = None

    @property
    def default(self) -> DriverBase:
        if self._default is None:
            self._default = _create_driver()
        return self._default

    @property
    def active(self) -> DriverBase:
        if self._active is None:
            self._active = self.default
        return self._active

    def set_active(self, driver: DriverBase) -> None:
        self._active = driver

    def reset_active(self) -> None:
        self._active = self.default


driver = DriverConfig()
<|endoftext|>from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sysconfig
import tempfile
import logging
from pathlib import Path

from types import ModuleType

from .cache import get_cache_manager
from .. import knobs


@functools.lru_cache()
def _find_compiler(language: str) -> str:
    if language == "c":
        cc = os.environ.get("CC")
        if cc is not None:
            return cc
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is not None:
            return cc
        raise RuntimeError(
            "Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.")

    assert language == "c++"
    cxx = os.environ.get("CXX")
    if cxx is not None:
        return cxx

    clangxx = shutil.which("clang++")
    gxx = shutil.which("g++")
    cxx = gxx if gxx is not None else clangxx
    if cxx is not None:
        return cxx

    raise RuntimeError(
        "Failed to find C++ compiler. Please specify via CXX environment variable or set triton.knobs.build.impl.")


def _language_from_filename(source_name: str) -> str:
    ext = Path(source_name).suffix
    if ext == ".c":
        return "c"
    if ext in {".cc", ".cpp", ".cxx"}:
        return "c++"
