        schema = _get_schema(func, node_args)

    # Create the replacement node
    with module.graph.inserting_before(node):
        new_node = module.graph.call_function(func, tuple(node_args), node.kwargs)

    # Update getitem nodes that extract outputs from with_effects
    for user in list(node.users.keys()):
        if user.target is not operator.getitem:
            raise AssertionError(
                f"Expected user target to be operator.getitem, but got {user.target}"
            )
        # getitem(with_effects, 0) is the token node
        if user.args[1] == 0:
            for user_user in list(user.users.keys()):
                if user_user.op == "output":
                    output_tokens.append(user)

    # Copy metadata from old node to new node
    for k, v in node.meta.items():
        new_node.meta[k] = v
        if k == "unbacked_bindings":
            # Remove the extra layer for effect token
            old_bindings = new_node.meta[k]
            new_bindings = {
                k: path[1:] if path else path for k, path in old_bindings.items()
            }
            new_node.meta[k] = new_bindings

    # Fix up the getitem nodes based on return count
    if len(schema.returns) == 1:
        # Single return: replace getitem(with_effects, 1) with the node itself
        for user in list(node.users.keys()):
            if user.args[1] == 1:
                user.replace_all_uses_with(new_node)
        new_node.meta["val"] = node.meta["val"][1]
    elif len(schema.returns) > 1:
        # Multiple returns: shift getitem indices down by 1
        for user in list(node.users.keys()):
            if user.args[1] >= 1:
                user.args = (new_node, user.args[1] - 1)
        new_node.meta["val"] = node.meta["val"][1:]
    else:
        # No returns
        if len(schema.returns) != 0:
            raise AssertionError(
                f"Expected schema.returns to be empty, but got {len(schema.returns)} returns"
            )
        if len(new_node.users) != 0:
            raise AssertionError(
                f"Expected new_node to have no users, but got {len(new_node.users)} users"
            )
        new_node.meta["val"] = None


def _replace_invoke_subgraph_node(node, module, output_tokens, input_tokens):
    """Replace an invoke_subgraph node to remove the token argument."""
    if node.args[0].op != "get_attr":
        raise AssertionError(
            f"Expected node.args[0].op to be 'get_attr', but got {node.args[0].op}"
        )
    submod = getattr(module, node.args[0].target)
    if not submod.meta.get("has_with_effects", False):
        return

    # Remove token from inputs
    subgraph, identifier, token, *operands = node.args
    node.args = (subgraph, identifier, *operands)
    if token.op == "placeholder":
        input_tokens.append(token)

    # Update getitem nodes to account for removed token output
    for user in list(node.users.keys()):
        if user.args[1] >= 1:
            user.args = (node, user.args[1] - 1)
        elif user.args[1] == 0:
            for user_user in list(user.users.keys()):
                if user_user.op == "output":
                    output_tokens.append(user)


def _remove_effect_tokens(ep: ExportedProgram) -> ExportedProgram:
    """
    Removes the existence of tokens from the exported program, including:
    - Removes the input and output tokens
    - Replaces with_effects(token, func, args) with just func(args)

    This function does an inplace modification on the given ExportedProgram.
    """
    inputs_to_lifted_custom_objs = ep.graph_signature.inputs_to_lifted_custom_objs

    # mark submodules with effects as having effects. This will be used in the following pass to remove effects from subgraphs
    for _, module in ep.graph_module.named_modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        with_effect_nodes = [
            node for node in module.graph.nodes if node.target is with_effects
        ]
        if len(with_effect_nodes) > 0:
            module.meta["has_with_effects"] = True

    # Process each module with the replace hook to ensure graph signature is updated
    with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
        for _, module in ep.graph_module.named_modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue

            input_tokens = []
            output_tokens = []

            # Process with_effects and invoke_subgraph nodes
            for node in module.graph.nodes:
                if node.target is with_effects:
                    _replace_with_effects_node(
                        node,
                        ep,
                        inputs_to_lifted_custom_objs,
                        output_tokens,
                        input_tokens,
                        module,
                    )
                elif node.target is torch.ops.higher_order.invoke_subgraph:
                    _replace_invoke_subgraph_node(
                        node, module, output_tokens, input_tokens
                    )

            # Remove tokens from the output node
            if len(output_tokens) > 0:
                output_node = next(reversed(module.graph.find_nodes(op="output")))
                output_args = output_node.args[0]
                if len(output_args) < len(output_tokens):
                    raise AssertionError(
                        f"{output_args} output arguments found\n"
                        f"{output_tokens} output tokens found\n"
                        f"{module.graph}"
                    )
                output_node.args = (tuple(output_args[len(output_tokens) :]),)

            module.graph.eliminate_dead_code()

            # Remove tokens from the input placeholders
            for node in module.graph.nodes:
                if node.op == "placeholder" and node in input_tokens:
                    module.graph.erase_node(node)

            module.recompile()

    num_tokens: int = 0
    input_token_names: list[str] = []
    new_input_specs: list[InputSpec] = []
    for inp in ep.graph_signature.input_specs:
        if inp.kind == InputKind.TOKEN:
            num_tokens += 1
            if not isinstance(inp.arg, TokenArgument):
                raise AssertionError(
                    f"Expected inp.arg to be a TokenArgument, but got {type(inp.arg)}"
                )
            input_token_names.append(inp.arg.name)
        else:
            new_input_specs.append(inp)

    num_out_tokens: int = 0
    new_output_specs: list[OutputSpec] = []
    output_token_names: list[OutputSpec] = []
    for out in ep.graph_signature.output_specs:
        if out.kind == OutputKind.TOKEN:
            num_out_tokens += 1
            output_token_names.append(out.arg.name)
        else:
            new_output_specs.append(out)

    # Update graph signature
    ep.graph_signature.input_specs = new_input_specs
    ep.graph_signature.output_specs = new_output_specs

    if num_tokens != num_out_tokens:
        raise AssertionError(
            f"Number of input tokens ({num_tokens}) does not match output tokens ({num_out_tokens})"
        )

    return ep
<|endoftext|>import gc
import types
import typing
import weakref
from typing_extensions import TypeIs

from torch.fx.experimental.symbolic_shapes import TrackedFake


"""
These functions are used to detect potential fake tensor leakage when using PT2 export.
See NOTE [export non-strict fake tensor leak detection]

There are some complications that made this logic overly complicated:
1) Python 3.10 and Python 3.12 have different ways of implementing referrer so
   we need to account for whether it is ref.__dict__ or the real ref object

2) There are some internal PT2 references to fake tensors like `TrackedFake`
3) closures, generators, and bound methods can hold fake tensors.
4) global object can hold onto a fake tensor

In general, these utils are our last resort to detect fake tensors. if the leak happens
within the model attributes, we have a separate mechanism to detect. This tool relies a bit
on garbage collector internal details, so I think it is unsafe to turn on by default, hence
this tool should be used as debugging tool.
"""


# Things we never want to flag as leaks
_SKIP_TYPES = (
    types.FrameType,
    types.ModuleType,
)


def _is_globals_or_locals(obj: typing.Any) -> bool:
    # These comparisons only make sense within this frame; still cheap to check.
    return obj is globals() or obj is locals()


def _is_tracked_fake(obj: typing.Any) -> TypeIs[TrackedFake]:
    return isinstance(obj, TrackedFake)


def _is_gm_meta_like_dict(d: dict, o: typing.Any) -> bool:
    # Hope gm.meta was a custom dict we can assert on
    return d.get("val") is o


def _dict_is_attr_of_tracked_fake(d: dict) -> bool:
    """
    Python 3.10 quirk: sometimes the referrer is obj.__dict__ instead of obj.
    Check if this dict is exactly the __dict__ of a TrackedFake.
    """
    for parent in gc.get_referrers(d):
        if (
            hasattr(parent, "__dict__")
            and parent.__dict__ is d
            and _is_tracked_fake(parent)
        ):
            return True
    return False


def find_legit_leaks_from_referrers(active_fakes: weakref.WeakSet) -> weakref.WeakSet:
    legit_leak: weakref.WeakSet = weakref.WeakSet()

    # This is so that we don't falsely flag generator to be holding fake tensor
    fake_list = list(active_fakes)
    fake_list_id = id(fake_list)

    for act in fake_list:
        # Track by id to avoid processing duplicate referrers
        seen = set()
        # Assume it's a leak unless we find only ignorable referrers
        flagged = False

        for r in gc.get_referrers(act):
            rid = id(r)
            if rid in seen:
                continue
            seen.add(rid)

            # Skip our own fake_list
            if rid == fake_list_id:
                continue

            # Fast-path: skip obvious non-owners
            if _is_globals_or_locals(r):
                continue
            if isinstance(r, _SKIP_TYPES):
                continue
            if _is_tracked_fake(r):
                # TrackedFake should be ignored
                continue

            # Handle dicts carefully (Python 3.10 sometimes shows __dict__)
            if isinstance(r, dict):
                if _is_gm_meta_like_dict(r, act):
                    continue
                if _dict_is_attr_of_tracked_fake(r):
                    continue
                flagged = True
                break

            # Any other referrer we don't explicitly whitelist counts as a leak
            flagged = True
            break

        if flagged:
            legit_leak.add(act)

    return legit_leak
<|endoftext|>from typing import Union

import torch
import torch.utils._pytree as pytree
from torch.export.exported_program import ExportedProgram


__all__ = ["move_to_device_pass"]


def move_to_device_pass(
    ep: ExportedProgram, location: torch.device | str | dict[str, str]
) -> ExportedProgram:
    """
    Move the exported program to the given device.

    Args:
        ep (ExportedProgram): The exported program to move.
        location (Union[torch.device, str, Dict[str, str]]): The device to move the exported program to.
            If a string, it is interpreted as a device name.
            If a dict, it is interpreted as a mapping from
            the existing device to the intended one

    Returns:
        ExportedProgram: The moved exported program.
    """

    def _get_new_device(
        curr_device: torch.device,
        location: torch.device | str | dict[str, str],
    ) -> str:
        if isinstance(location, dict):
            if str(curr_device) in location:
                return location[str(curr_device)]
            else:
                return str(curr_device)
        else:
            return str(location)

    # move all the state_dict
    for k, v in ep.state_dict.items():
        if isinstance(v, torch.nn.Parameter):
            ep._state_dict[k] = torch.nn.Parameter(
                v.to(_get_new_device(v.device, location)),
                v.requires_grad,
            )
        else:
            ep._state_dict[k] = v.to(_get_new_device(v.device, location))

    # move all the constants
    for k, v in ep.constants.items():
        if isinstance(v, torch.Tensor):
            ep._constants[k] = v.to(_get_new_device(v.device, location))

    # move example_inputs if they exist
    if ep.example_inputs is not None:
        args, kwargs = ep.example_inputs
        moved_args = pytree.tree_map_only(
            torch.Tensor,
            lambda tensor: tensor.to(_get_new_device(tensor.device, location)),
            args,
        )
        moved_kwargs = pytree.tree_map_only(
            torch.Tensor,
            lambda tensor: tensor.to(_get_new_device(tensor.device, location)),
            kwargs,
        )
        ep._example_inputs = (moved_args, moved_kwargs)

    for m in ep.graph_module.modules():
        if isinstance(m, torch.fx.GraphModule):
            for node in m.graph.nodes:
                # move all the nodes kwargs with burnt-in device
                if "device" in node.kwargs:
                    kwargs = node.kwargs.copy()
                    kwargs["device"] = _get_new_device(kwargs["device"], location)
                    node.kwargs = kwargs

                if (
                    node.op == "call_function"
                    and node.target is torch.ops.aten.to.device
                ):
                    args = list(node.args)
                    # pyrefly: ignore [unsupported-operation]
                    args[1] = _get_new_device(args[1], location)
                    node.args = tuple(args)

                # move all the tensor metadata
                node.meta["val"] = pytree.tree_map(
                    lambda v: v.to(_get_new_device(v.device, location))
                    if isinstance(v, torch.Tensor)
                    else v,
                    node.meta.get("val"),
                )

    ep.validate()
    return ep
<|endoftext|>import glob
import io
import json
import logging
import os
import pickle
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, IO, TYPE_CHECKING, TypeAlias
from typing_extensions import TypeIs

import torch
import torch.utils._pytree as pytree
from torch._export.serde import schema
from torch._export.serde.serialize import (
    _dataclass_to_dict,
    _dict_to_dataclass,
    deserialize_device,
    deserialize_scalar_type,
    deserialize_size,
    deserialize_storage_offset,
    deserialize_stride,
    ExportedProgramDeserializer,
    serialize,
    serialize_tensor_meta,
    SerializedArtifact,
)
from torch._inductor.cpp_builder import normalize_path_separator
from torch._library.opaque_object import is_custom_class_obj
from torch._subclasses.fake_tensor import FakeTensor, is_fake_tensor
from torch.export import ExportedProgram
from torch.export._tree_utils import reorder_kwargs
from torch.export.pt2_archive._package_weights import (
    get_complete_tensor,
    group_weights,
    TensorProperties,
    Weights,
)
from torch.export.pt2_archive.constants import (
    AOTINDUCTOR_DIR,
    ARCHIVE_FORMAT_PATH,
    ARCHIVE_FORMAT_VALUE,
    ARCHIVE_VERSION_PATH,
    ARCHIVE_VERSION_VALUE,
    CONSTANTS_CONFIG_FILENAME_FORMAT,
    CONSTANTS_DIR,
    CUSTOM_OBJ_FILENAME_PREFIX,
    EXECUTORCH_DIR,
    EXTRA_DIR,
    MODELS_DIR,
    MODELS_FILENAME_FORMAT,
    OPAQUE_OBJ_FILENAME_PREFIX,
    SAMPLE_INPUTS_FILENAME_FORMAT,
    TENSOR_CONSTANT_FILENAME_PREFIX,
    WEIGHT_FILENAME_PREFIX,
    WEIGHTS_CONFIG_FILENAME_FORMAT,
    WEIGHTS_DIR,
)
from torch.types import FileLike


if TYPE_CHECKING:
    from torch.utils._ordered_set import OrderedSet


DEFAULT_PICKLE_PROTOCOL = 2
AOTI_FILES: TypeAlias = list[str | Weights] | dict[str, list[str | Weights]]


logger: logging.Logger = logging.getLogger(__name__)


def is_pt2_package(serialized_model: bytes | str) -> bool:
    """
    Check if the serialized model is a PT2 Archive package.
    """
    try:
        with zipfile.ZipFile(
            io.BytesIO(serialized_model)
            if isinstance(serialized_model, bytes)
            else serialized_model
        ) as zip_reader:
            root_folder = zip_reader.namelist()[0].split(os.path.sep)[0]
            archive_format_path = f"{root_folder}/{ARCHIVE_FORMAT_PATH}"
            if archive_format_path in zip_reader.namelist():
                return zip_reader.read(archive_format_path) == b"pt2"
    except Exception:
        logger.info("Model is not a PT2 package")
    return False


class PT2ArchiveWriter:
    """
    Context manager for writing a PT2 archive.
    """

    def __init__(self, archive_path_or_buffer: FileLike):
        if isinstance(archive_path_or_buffer, str):
            archive_path_or_buffer = normalize_path_separator(archive_path_or_buffer)
        self.archive_file = torch._C.PyTorchFileWriter(archive_path_or_buffer)  # type: ignore[arg-type]
        # NOTICE: version here is different from the archive_version
        # this is the version of zip file format, which is used by PyTorchFileWriter, which write to /.data/version
        # archive_version is the version of the PT2 archive spec, which write to /archive_version
        self.archive_file.set_min_version(6)

    def __enter__(self) -> "PT2ArchiveWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        if not self.has_record(ARCHIVE_FORMAT_PATH):
            self.write_string(ARCHIVE_FORMAT_PATH, ARCHIVE_FORMAT_VALUE)

        if not self.has_record(ARCHIVE_VERSION_PATH):
            self.write_string(ARCHIVE_VERSION_PATH, ARCHIVE_VERSION_VALUE)

        self.close()

    def has_record(self, name: str) -> bool:
        """
        Check if a record exists in the archive.
        """
        return name in self.archive_file.get_all_written_records()

    def count_prefix(self, prefix: str) -> int:
        """
        Count the number of records that start with a given prefix.
        """
        return sum(
            1
            for record in self.archive_file.get_all_written_records()
            if record.startswith(prefix)
        )

    def write_bytes(self, name: str, data: bytes) -> None:
        """
        Write a bytes object to the archive.
        name: The destination file inside the archive.
        data: The bytes object to write.
        """
        if not isinstance(data, bytes):
            raise AssertionError(f"Expected bytes but got {type(data)}")
        self.archive_file.write_record(name, data, len(data))

    def write_string(self, name: str, data: str) -> None:
        """
        Write a string object to the archive.
        name: The destination file inside the archive.
        data: The string object to write.
        """
        if not isinstance(data, str):
            raise AssertionError(f"Expected string but got {type(data)}")
        data_bytes = data.encode()
        self.write_bytes(name, data_bytes)

    def write_file(self, name: str, file_path: str) -> None:
        """
        Copy a file into the archive.
        name: The destination file inside the archive.
        file_path: The source file on disk.
        """
        if not os.path.isfile(file_path):
            raise AssertionError(f"{file_path} is not a valid file path")

        with open(file_path, "rb") as f:
            file_bytes = f.read()
            self.write_bytes(name, file_bytes)

    def write_folder(self, archive_dir: str, folder_dir: str) -> None:
        """
        Copy a folder into the archive.
        archive_dir: The destination folder inside the archive.
        folder_dir: The source folder on disk.
        """
        if not os.path.isdir(folder_dir):
            raise AssertionError(f"{folder_dir} is not a valid directory path")

        file_paths = filter(
            os.path.isfile, glob.glob(f"{folder_dir}/**", recursive=True)
        )
        for file_path in file_paths:
            # pyrefly: ignore [no-matching-overload]
            filename = os.path.relpath(file_path, folder_dir)
            archive_path = os.path.join(archive_dir, filename)
            # pyrefly: ignore [bad-argument-type]
            self.write_file(archive_path, file_path)

    def close(self) -> None:
        """
        Close the archive.
        """
        self.archive_file.write_end_of_file()


class PT2ArchiveReader:
    """
    Context manager for reading a PT2 archive.
    """

    def __init__(self, archive_path_or_buffer: FileLike):
        if isinstance(archive_path_or_buffer, str):
            archive_path_or_buffer = normalize_path_separator(archive_path_or_buffer)
        self.archive_file = torch._C.PyTorchFileReader(archive_path_or_buffer)  # type: ignore[arg-type]
        if self.read_string(ARCHIVE_FORMAT_PATH) != ARCHIVE_FORMAT_VALUE:
            raise AssertionError("Invalid archive format")

    def __enter__(self) -> "PT2ArchiveReader":
        return self

    def __exit__(self, *args: Any) -> None:
        # torch._C.PyTorchFileReader doesn't have a close method
        pass

    def read_bytes(self, name: str) -> bytes:
        """
        Read a bytes object from the archive.
        name: The source file inside the archive.
        """
        return self.archive_file.get_record(name)

    def read_string(self, name: str) -> str:
        """
        Read a string object from the archive.
        name: The source file inside the archive.
        """
        data = self.read_bytes(name)
        return data.decode()

    def archive_version(self) -> int:
        """
        Get the archive version.
        """
        try:
            archive_version = self.read_string(ARCHIVE_VERSION_PATH)
        except Exception:
            # if archive_version is not found, it means the archive is older than version 0.
            # In this case, we assume the archive is version 0.
            archive_version = "0"

        return int(archive_version)

    def get_file_names(self) -> list[str]:
        """
        Get the file names in the archive.
        """
        return self.archive_file.get_all_records()


is_pt2_package.__module__ = "torch.export.pt2_archive"
PT2ArchiveWriter.__module__ = "torch.export.pt2_archive"
PT2ArchiveReader.__module__ = "torch.export.pt2_archive"


def _package_aoti_files(
    archive_writer: PT2ArchiveWriter,
    aoti_files: AOTI_FILES | None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> None:
    if aoti_files is None:
        return

    if isinstance(aoti_files, list):
        aoti_files = {"model": aoti_files}

    if not isinstance(aoti_files, dict):
        raise AssertionError(
            f"Expected aoti_files to be a dict, but got {type(aoti_files)}"
        )

    all_weights: dict[str, Weights] = {}  # model_name -> weight
    weights_configs: dict[
        str, dict[str, Any]
    ] = {}  # model_name -> (weight_name -> (filename, shape, stride, offset))

    for model_name, files in aoti_files.items():
        weights_configs[model_name] = {}

        for file in files:
            if file == "":
                continue

            if isinstance(file, Weights):
                all_weights[model_name] = file
                continue

            # Note: Previously we rejected multiple .so cases. But Triton CPU AOTI
            # has multiple .so files per model (wrapper, launcher, kernel).

            filename = os.path.basename(file)
            if filename.startswith(CUSTOM_OBJ_FILENAME_PREFIX):
                new_filepath = os.path.join(CONSTANTS_DIR, filename)
            else:
                new_filepath = os.path.join(AOTINDUCTOR_DIR, model_name, filename)
            logger.debug(
                "Saving AOTI generated file %s to archive in %s", file, new_filepath
            )
            archive_writer.write_file(
                str(new_filepath),
                file,
            )

    if len(all_weights) > 0:
        # Dedup weights
        grouped_tensors: list[OrderedSet[tuple[str, str]]] = group_weights(all_weights)
        for idx, group in enumerate(grouped_tensors):
            filename = f"{WEIGHT_FILENAME_PREFIX}{idx}"
            complete_tensor = get_complete_tensor(group, all_weights)
            buffer = io.BytesIO()
            torch.save(complete_tensor, buffer, pickle_protocol=pickle_protocol)
            archive_writer.write_bytes(
                os.path.join(WEIGHTS_DIR, filename), buffer.getvalue()
            )
            for model_name, weight_name in group:
                _, w_property = all_weights[model_name].get_weight(weight_name)
                weights_configs[model_name][weight_name] = (
                    filename,
                    w_property.shape,
                    w_property.stride,
                    w_property.offset,
                )

        for model_name, weights_config in weights_configs.items():
            archive_writer.write_string(
                os.path.join(AOTINDUCTOR_DIR, model_name, "weights_config.json"),
                json.dumps(weights_config),
            )
            logger.debug("packaging weights_config for model %s", model_name)
            logger.debug(weights_config)


def _is_fake_tensor(t: torch.Tensor) -> TypeIs[FakeTensor]:
    return is_fake_tensor(t)


def _is_tensor_subclass(t: torch.Tensor) -> bool:
    return isinstance(t, torch.Tensor) and type(t.data) is not torch.Tensor


def _get_raw_tensor_bytes(value: torch.Tensor) -> bytes:
    """
    Get the raw bytes of a tensor. This is used to save the tensor in pt2 archive.
    """
    # NOTE: don't chain .cpu() with .data_ptr(). If an HtoD copy needs to be
    # performed, the CPU copy needs to be kept alive when its underlying
    # memory is accessed.
    import ctypes

    if _is_fake_tensor(value):
        value_bytes = b""
    elif value.data_ptr():
        cpu_tensor = value.cpu()
        value_untyped_storage = cpu_tensor.untyped_storage()
        # we store the raw bytes the untyped storage. Tensor metadata is stored separately
        value_bytes = bytes(
            ctypes.cast(
                value_untyped_storage.data_ptr(),
                ctypes.POINTER(ctypes.c_ubyte * value_untyped_storage.size()),
            ).contents
        )
    else:
        # for empty tensor
        value_bytes = b""
    return value_bytes


def _should_use_pickle(t: torch.Tensor) -> bool:
    return _is_tensor_subclass(t) and not _is_fake_tensor(t)


def _save_pickled_tensors(
    pickled_items: list[tuple[str, torch.Tensor]],
    archive_writer: PT2ArchiveWriter,
    config: dict[str, schema.PayloadMeta],
    directory: str,
    filename_prefix: str,
    idx: int,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> int:
    """Save pickled tensors and update config. Returns updated index."""
    for item_fqn, tensor in pickled_items:
        path_name = f"{filename_prefix}{idx}"
        archive_path = os.path.join(directory, path_name)
        buffer = io.BytesIO()
        torch.save(tensor, buffer, pickle_protocol=pickle_protocol)
        archive_writer.write_bytes(archive_path, buffer.getvalue())

        config[item_fqn] = schema.PayloadMeta(
            path_name=path_name,
            is_param=isinstance(tensor, torch.nn.Parameter),
            use_pickle=True,
            tensor_meta=serialize_tensor_meta(tensor),
        )
        idx += 1
    return idx


def _save_raw_tensors(
    raw_items: dict[str, tuple[torch.Tensor, TensorProperties]],
    model_name: str,
    archive_writer: PT2ArchiveWriter,
    config: dict[str, schema.PayloadMeta],
    directory: str,
    filename_prefix: str,
    idx: int,
) -> int:
    """Save deduplicated raw tensor bytes and update config. Returns updated index."""
    if not raw_items:
        return idx

    weights_dict = {model_name: Weights(raw_items)}
    storage_groups = group_weights(weights_dict)

    for group in storage_groups:
        # Find the complete tensor that covers all others in this storage group
        complete_tensor = get_complete_tensor(group, weights_dict)

        path_name = f"{filename_prefix}{idx}"
        archive_path = os.path.join(directory, path_name)
        tensor_bytes = _get_raw_tensor_bytes(complete_tensor)
        archive_writer.write_bytes(archive_path, tensor_bytes)
        idx += 1

        for _, item_fqn in group:
            tensor, _ = weights_dict[model_name].get_weight(item_fqn)
            config[item_fqn] = schema.PayloadMeta(
                path_name=path_name,
                is_param=isinstance(tensor, torch.nn.Parameter),
                use_pickle=False,
                tensor_meta=serialize_tensor_meta(tensor),
            )

    return idx


def _package_state_dict(
    model_name: str,
    exported_program: ExportedProgram,
    archive_writer: PT2ArchiveWriter,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> schema.PayloadConfig:
    weights_config: dict[str, schema.PayloadMeta] = {}

    pickled_weights: list[tuple[str, torch.Tensor]] = []
    raw_weights: dict[str, tuple[torch.Tensor, TensorProperties]] = {}

    # Categorize weights
    for weight_fqn, weight_tensor in exported_program.state_dict.items():
        if not isinstance(weight_tensor, torch.Tensor):
            raise AssertionError("only torch.Tensor is allowed in state_dict")
        if _should_use_pickle(weight_tensor):
            pickled_weights.append((weight_fqn, weight_tensor))
        else:
            raw_weights[weight_fqn] = (weight_tensor, TensorProperties(weight_tensor))

    idx = archive_writer.count_prefix(os.path.join(WEIGHTS_DIR, WEIGHT_FILENAME_PREFIX))

    # Save weights in pickle format
    idx = _save_pickled_tensors(
        pickled_weights,
        archive_writer,
        weights_config,
        WEIGHTS_DIR,
        WEIGHT_FILENAME_PREFIX,
        idx,
        pickle_protocol,
    )

    # Save weights in raw bytes format
    _save_raw_tensors(
        raw_weights,
        model_name,
        archive_writer,
        weights_config,
        WEIGHTS_DIR,
        WEIGHT_FILENAME_PREFIX,
        idx,
    )

    return schema.PayloadConfig(config=weights_config)


def _package_constants(
    model_name: str,
    exported_program: ExportedProgram,
    archive_writer: PT2ArchiveWriter,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> schema.PayloadConfig:
    constants_config: dict[str, schema.PayloadMeta] = {}

    pickled_constants: list[tuple[str, torch.Tensor]] = []
    raw_constants: dict[str, tuple[torch.Tensor, TensorProperties]] = {}
    custom_objects: list[tuple[str, Any]] = []

    # Categorize constants
    for constant_fqn, constant in exported_program.constants.items():
        if isinstance(constant, torch.Tensor):
            if _should_use_pickle(constant):
                pickled_constants.append((constant_fqn, constant))
            else:
                raw_constants[constant_fqn] = (constant, TensorProperties(constant))

        elif isinstance(constant, torch._C.ScriptObject) or is_custom_class_obj(
            constant
        ):
            custom_objects.append((constant_fqn, constant))

        else:
            raise RuntimeError(f"Unsupported constant type: {type(constant)}")

    tensor_idx = archive_writer.count_prefix(
        os.path.join(CONSTANTS_DIR, TENSOR_CONSTANT_FILENAME_PREFIX)
    )
    custom_obj_idx = archive_writer.count_prefix(
        os.path.join(CONSTANTS_DIR, CUSTOM_OBJ_FILENAME_PREFIX)
    )
    opaque_obj_idx = archive_writer.count_prefix(
        os.path.join(CONSTANTS_DIR, OPAQUE_OBJ_FILENAME_PREFIX)
    )

    # Save constants in pickle format
    tensor_idx = _save_pickled_tensors(
        pickled_constants,
        archive_writer,
        constants_config,
        CONSTANTS_DIR,
        TENSOR_CONSTANT_FILENAME_PREFIX,
        tensor_idx,
        pickle_protocol,
    )

    # Save constants in raw bytes format
    _save_raw_tensors(
        raw_constants,
        model_name,
        archive_writer,
        constants_config,
        CONSTANTS_DIR,
        TENSOR_CONSTANT_FILENAME_PREFIX,
        tensor_idx,
    )

    # ScriptObjects use the torchbind-aware pickler (zip-archive format,
    # readable from C++ at load time). Opaque values use standard Python
    # pickle. Distinct filename prefixes act as the format discriminator
    # for the reader.
    for constant_fqn, constant in custom_objects:
        if isinstance(constant, torch._C.ScriptObject):
            path_name = f"{CUSTOM_OBJ_FILENAME_PREFIX}{custom_obj_idx}"
            obj_bytes = torch._C._pickle_save(constant)
            custom_obj_idx += 1
        else:
            path_name = f"{OPAQUE_OBJ_FILENAME_PREFIX}{opaque_obj_idx}"
            obj_bytes = pickle.dumps(constant, protocol=pickle_protocol)
            opaque_obj_idx += 1
        archive_path = os.path.join(CONSTANTS_DIR, path_name)
        archive_writer.write_bytes(archive_path, obj_bytes)

        constants_config[constant_fqn] = schema.PayloadMeta(
            path_name=path_name,
            is_param=False,
            use_pickle=True,
            tensor_meta=None,
        )

    return schema.PayloadConfig(config=constants_config)


def _package_payload_config(
    archive_writer: PT2ArchiveWriter,
    payload_config: schema.PayloadConfig,
    config_file: str,
) -> None:
    """
    Save the payload config as json file in the archive.
    """
    archive_writer.write_string(
        config_file, json.dumps(_dataclass_to_dict(payload_config))
    )


def _package_exported_programs(
    archive_writer: PT2ArchiveWriter,
    exported_programs: ExportedProgram | dict[str, ExportedProgram] | None,
    opset_version: dict[str, int] | None = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> None:
    if exported_programs is None:
        return

    if isinstance(exported_programs, ExportedProgram):
        exported_programs = {"model": exported_programs}

    if not isinstance(exported_programs, dict):
        raise AssertionError(
            f"Expected exported_programs to be a dict, but got {type(exported_programs)}"
        )

    for model_name, ep in exported_programs.items():
        weights_config = _package_state_dict(
            model_name, ep, archive_writer, pickle_protocol
        )
        weights_config_file = WEIGHTS_CONFIG_FILENAME_FORMAT.format(model_name)
        _package_payload_config(archive_writer, weights_config, weights_config_file)

        constants_config = _package_constants(
            model_name, ep, archive_writer, pickle_protocol
        )
        constants_config_file = CONSTANTS_CONFIG_FILENAME_FORMAT.format(model_name)
        _package_payload_config(archive_writer, constants_config, constants_config_file)

        artifact: SerializedArtifact = serialize(
            ep,
            opset_version,
            pickle_protocol,
            serialize_state_dict=False,
            serialize_constants=False,
        )

        archive_writer.write_bytes(
            MODELS_FILENAME_FORMAT.format(model_name), artifact.exported_program
        )
        archive_writer.write_bytes(
            SAMPLE_INPUTS_FILENAME_FORMAT.format(model_name),
            artifact.example_inputs,
        )


def _package_extra_files(
    archive_writer: PT2ArchiveWriter, extra_files: dict[str, Any] | None
) -> None:
    if extra_files is None:
        return

    for extra_file_name, content in extra_files.items():
        archive_writer.write_string(f"{EXTRA_DIR}{extra_file_name}", content)


def _package_executorch_files(
    archive_writer: PT2ArchiveWriter, executorch_files: dict[str, bytes] | None
) -> None:
    if executorch_files is None:
        return

    for file_name, content in executorch_files.items():
        archive_writer.write_bytes(f"{EXECUTORCH_DIR}{file_name}", content)


def package_pt2(
    f: FileLike,
    *,
    exported_programs: ExportedProgram | dict[str, ExportedProgram] | None = None,
    aoti_files: AOTI_FILES | None = None,
    extra_files: dict[str, Any] | None = None,
    opset_version: dict[str, int] | None = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
    executorch_files: dict[str, bytes] | None = None,
) -> FileLike:
    r"""
    Saves the artifacts to a PT2Archive format. The artifact can then be loaded
    using ``load_pt2``.

    Args:
        f (str | os.PathLike[str] | IO[bytes]): A file-like object (has to
         implement write and flush) or a string containing a file name.

        exported_programs (Union[ExportedProgram, dict[str, ExportedProgram]]):
         The exported program to save, or a dictionary mapping model name to an
         exported program to save. The exported program will be saved under
         models/\*.json. If only one ExportedProgram is specified, this will
         automatically be named "model".

        aoti_files (Union[list[str], dict[str, list[str]]]): A list of files
         generated by AOTInductor via
         ``torch._inductor.aot_compile(..., {"aot_inductor.package": True})``,
         or a dictionary mapping model name to its AOTInductor generated files.
         If only one set of files is specified, this will automatically be named
         "model".

        extra_files (Optional[Dict[str, Any]]): Map from filename to contents
         which will be stored as part of the pt2.

        opset_version (Optional[Dict[str, int]]): A map of opset names
         to the version of this opset

        pickle_protocol: can be specified to override the default protocol

        executorch_files (Optional[dict[str, bytes]]): Optional executorch
         artifacts to save.

    """
    if exported_programs is None and aoti_files is None and extra_files is None:
        raise AssertionError(
            "No value passed in for `exported_programs`, `aoti_files`, and "
            "`extra_files`, implying that you do not plan on saving anything."
        )

    if not (
        (isinstance(f, (io.IOBase, IO)) and f.writable() and f.seekable())
        or (isinstance(f, (str, os.PathLike)) and os.fspath(f).endswith(".pt2"))
        or (isinstance(f, tempfile._TemporaryFileWrapper) and f.name.endswith(".pt2"))
    ):
        # TODO: turn this into an error
        logger.warning(
            "Expect archive file to be a file ending in .pt2, or is a buffer. "
            "Instead got {%s}",
            f,
        )

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    # pyrefly: ignore [bad-argument-type]
    with PT2ArchiveWriter(f) as archive_writer:
        _package_exported_programs(
            archive_writer, exported_programs, pickle_protocol=pickle_protocol
        )
        _package_aoti_files(
            archive_writer,
            aoti_files,
            pickle_protocol=pickle_protocol,
        )
        _package_extra_files(archive_writer, extra_files)
        _package_executorch_files(archive_writer, executorch_files)

    if isinstance(f, (io.IOBase, IO)):
        f.seek(0)
    # pyrefly: ignore [bad-return]
    return f


class AOTICompiledModel:
    """
    Callable AOT Inductor loaded model from a .pt2
    """

    def __init__(self, loader: torch._C._aoti.AOTIModelPackageLoader) -> None:
        self.loader = loader

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        call_spec = self.loader.get_call_spec()
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
        flat_outputs = self.loader.boxed_run(flat_inputs)
        return pytree.tree_unflatten(flat_outputs, out_spec)

    def get_metadata(self) -> dict[str, str]:
        return self.loader.get_metadata()

    def load_constants(
        self,
        constants_map: dict[str, torch.Tensor],
        *,
        check_full_update: bool,
        user_managed: bool = False,
        allow_h2d_copy: bool = False,
    ) -> None:
        """
        Given a mapping of constant fqns to tensors, load the constants into the model.
        You can use ``get_constant_fqns`` to get the list of constant fqns that
        are needed in the compiled model.

        Args:
            constants_map: A mapping of constant fqns to tensors.
            check_full_update: Whether to add check to see if all the constants
            are updated and have values.
            user_managed: If True, the loader stores the tensor pointers
            directly; the caller must keep them alive.
            allow_h2d_copy: If True, CPU tensors are silently copied to the
            model's device. Useful for loading a CPU ``state_dict()`` into a
            non-CPU model. Incompatible with ``user_managed``.
        """
        self.loader.load_constants(
            constants_map, False, check_full_update, user_managed, allow_h2d_copy
        )

    def get_constant_fqns(self) -> list[str]:
        return self.loader.get_constant_fqns()

    def __deepcopy__(self, memo: dict[Any, Any] | None) -> "AOTICompiledModel":
        logger.warning(
            "AOTICompiledModel deepcopy warning: AOTICompiledModel.loader is not deepcopied."
        )
        return AOTICompiledModel(self.loader)


@dataclass
class PT2ArchiveContents:
    exported_programs: dict[str, ExportedProgram]
    aoti_runners: dict[str, AOTICompiledModel]
    extra_files: dict[str, Any]


def _build_file_map(
    archive_reader: PT2ArchiveReader,
    config: schema.PayloadConfig,
    base_dir: str,
) -> dict[str, torch.Tensor]:
    """
    Build a map from file path to the payload in flat tensor format.
    """
    file_map: dict[str, torch.Tensor] = {}
    for payload_meta in config.config.values():
        # skip pickled objects
        if payload_meta.use_pickle:
            continue
        # skip files that already exist in the map
        if payload_meta.path_name in file_map:
            continue

        if payload_meta.tensor_meta is None:
            raise AssertionError("payload_meta.tensor_meta cannot be None")

        tensor_meta = payload_meta.tensor_meta
        dtype = deserialize_scalar_type(tensor_meta.dtype)
        device = deserialize_device(tensor_meta.device)
        record_name = os.path.join(base_dir, payload_meta.path_name)
        nbytes = archive_reader.archive_file.get_record_size(record_name)

        if nbytes == 0:
            size = deserialize_size(tensor_meta.sizes)
            tensor = torch.zeros(size, dtype=dtype, device=device)
            if tensor_meta.requires_grad:
                tensor.requires_grad_(True)
        else:
            element_size = torch._utils._element_size(dtype)
            if nbytes % element_size != 0:
                raise ValueError(
                    f"Record {record_name}: size {nbytes} is not a multiple of "
                    f"element size {element_size} for dtype {dtype}"
                )
            numel = nbytes // element_size
            raw = archive_reader.archive_file.get_storage_from_record(
                record_name, numel, dtype
            )
            # get_storage_from_record returns a tensor with empty DispatchKeySet,
            # so we wrap the storage in a proper CPU tensor via set_().
            tensor = torch.empty(0, dtype=dtype)
            tensor.set_(raw.untyped_storage(), 0, (numel,), (1,))
            if tensor_meta.requires_grad:
                tensor.requires_grad_(True)
            if device != torch.device("cpu"):
                tensor = tensor.to(device)

        file_map[payload_meta.path_name] = tensor

    return file_map


def _load_payload_config(
    archive_reader: PT2ArchiveReader,
    config_file: str,
) -> schema.PayloadConfig:
    """
    Load and parse a payload config from the archive.
    """
    return _dict_to_dataclass(
        schema.PayloadConfig,
        json.loads(archive_reader.read_string(config_file)),
    )


def _load_state_dict(
    archive_reader: PT2ArchiveReader,
    model_name: str,
) -> dict[str, torch.Tensor] | bytes:
    # Make it BC compatible with legacy weight files
    legacy_weights_file = f"{WEIGHTS_DIR}{model_name}.pt"
    if legacy_weights_file in archive_reader.get_file_names():
        logger.warning(
            "You are loading weight from the legacy format. "
            "Please generate a new pt2 file using torch.export.save()."
        )
        return archive_reader.read_bytes(legacy_weights_file)
    else:
        weights_config_file = WEIGHTS_CONFIG_FILENAME_FORMAT.format(model_name)
        if weights_config_file not in archive_reader.get_file_names():
            raise AssertionError(f"{weights_config_file} not found in PT2 archive")
        weights_config = _load_payload_config(archive_reader, weights_config_file)
        # construct the mapping from file name (e.g. weight_0) to flat weight payload
        state_dict_file_map = _build_file_map(
            archive_reader, weights_config, WEIGHTS_DIR
        )
        # chain the mapping weight FQN -> weight file name -> strided weight payload
        # so that the aliasing of weights is preserved
        state_dict: dict[str, torch.Tensor] = {}
        for weight_fqn, payload_meta in weights_config.config.items():
            if payload_meta.use_pickle:
                weight_bytes = archive_reader.read_bytes(
                    os.path.join(WEIGHTS_DIR, payload_meta.path_name)
                )
                state_dict[weight_fqn] = torch.load(
                    io.BytesIO(weight_bytes), weights_only=False
                )
            else:
                tensor_meta = payload_meta.tensor_meta
                if tensor_meta is None:
                    raise AssertionError(
                        "tensor_meta cannot be None for non-pickled weight"
                    )
                weight_tensor = torch.as_strided(
                    input=state_dict_file_map[payload_meta.path_name],
                    size=deserialize_size(tensor_meta.sizes),
                    stride=deserialize_stride(tensor_meta.strides),
                    storage_offset=deserialize_storage_offset(
                        tensor_meta.storage_offset
                    ),
                )
                if payload_meta.is_param:
                    state_dict[weight_fqn] = torch.nn.Parameter(
                        weight_tensor, requires_grad=tensor_meta.requires_grad
                    )
                else:
                    state_dict[weight_fqn] = weight_tensor

        return state_dict


def _load_constants(
    archive_reader: PT2ArchiveReader,
    model_name: str,
) -> dict[str, torch.Tensor] | bytes:
    # Make it BC compatible with legacy constant files
    legacy_constants_file = f"{CONSTANTS_DIR}{model_name}.pt"
    if legacy_constants_file in archive_reader.get_file_names():
        logger.warning(
            "You are loading constant from the legacy format. "
            "Please generate a new pt2 file using torch.export.save()."
        )
        return archive_reader.read_bytes(legacy_constants_file)
    else:
        constants_config_file = CONSTANTS_CONFIG_FILENAME_FORMAT.format(model_name)
        if constants_config_file not in archive_reader.get_file_names():
            raise AssertionError(f"{constants_config_file} not found in PT2 archive")
        constants_config = _load_payload_config(archive_reader, constants_config_file)
        # construct the mapping from file name (e.g. constant_0) to constant payload
        constant_file_map = _build_file_map(
            archive_reader, constants_config, CONSTANTS_DIR
        )
        # chain the mapping constant FQN -> constant file name -> strided constant payload
        # so that the aliasing of constants is preserved
        constants: dict[str, torch.Tensor] = {}
        for constant_fqn, payload_meta in constants_config.config.items():
            path_name = payload_meta.path_name
            if path_name.startswith(TENSOR_CONSTANT_FILENAME_PREFIX):
                if payload_meta.use_pickle:
                    constant_bytes = archive_reader.read_bytes(
                        os.path.join(CONSTANTS_DIR, path_name)
                    )
                    constants[constant_fqn] = torch.load(
                        io.BytesIO(constant_bytes), weights_only=False
                    )
                else:
                    tensor_meta = payload_meta.tensor_meta
                    if tensor_meta is None:
                        raise AssertionError(
                            "tensor_meta cannot be None for non-pickled constant"
                        )
                    constant_tensor = torch.as_strided(
                        input=constant_file_map[path_name],
                        size=deserialize_size(tensor_meta.sizes),
                        stride=deserialize_stride(tensor_meta.strides),
                        storage_offset=deserialize_storage_offset(
                            tensor_meta.storage_offset
                        ),
                    )
                    constants[constant_fqn] = constant_tensor

            elif path_name.startswith(CUSTOM_OBJ_FILENAME_PREFIX):
                constant_bytes = archive_reader.read_bytes(
                    os.path.join(CONSTANTS_DIR, path_name)
                )
                constants[constant_fqn] = torch._C._pickle_load_obj(constant_bytes)

            elif path_name.startswith(OPAQUE_OBJ_FILENAME_PREFIX):
                constant_bytes = archive_reader.read_bytes(
                    os.path.join(CONSTANTS_DIR, path_name)
                )
                constants[constant_fqn] = pickle.loads(constant_bytes)

            else:
                raise RuntimeError(f"Unsupported constant type: {path_name}")

        return constants


def _load_exported_programs(
    archive_reader: PT2ArchiveReader,
    file_names: list[str],
    expected_opset_version: dict[str, int] | None,
) -> dict[str, ExportedProgram]:
    exported_program_files = [
        file for file in file_names if file.startswith(MODELS_DIR)
    ]
    exported_programs = {}
    for file in exported_program_files:
        prefix, suffix = MODELS_FILENAME_FORMAT.split(
            "{}"
        )  # split "models/{}.json" into "models/" and "json"
        model_name = file[
            len(prefix) : -len(suffix)
        ]  # given "models/foo.json" we can now get "foo"

        sample_inputs_file = SAMPLE_INPUTS_FILENAME_FORMAT.format(model_name)
        serialized_sample_inputs = archive_reader.read_bytes(sample_inputs_file)

        from torch._export.serde.serialize import _bytes_to_dataclass

        exported_program_bytes = archive_reader.read_bytes(file)
        serialized_exported_program = _bytes_to_dataclass(
            schema.ExportedProgram, exported_program_bytes
        )
        state_dict = _load_state_dict(archive_reader, model_name)
        constants = _load_constants(archive_reader, model_name)

        ep = ExportedProgramDeserializer(expected_opset_version).deserialize(
            serialized_exported_program,
            state_dict,
            constants,
            serialized_sample_inputs,
        )

        exported_programs[model_name] = ep

    return exported_programs


def _load_extra_files(
    archive_reader: PT2ArchiveReader, file_names: list[str]
) -> dict[str, Any]:
    extra_files = [file for file in file_names if file.startswith(EXTRA_DIR)]

    extra_file_contents: dict[str, Any] = {}
    for file in extra_files:
        contents = archive_reader.read_string(file)
        extra_file_contents[file[len(EXTRA_DIR) :]] = contents

    return extra_file_contents


def _load_aoti(
    file: str,
    model_name: str,
    run_single_threaded: bool,
    num_runners: int,
    device_idx: int,
) -> AOTICompiledModel:
    loaded_metadata = torch._C._aoti.AOTIModelPackageLoader.load_metadata_from_package(  # type: ignore[attr-defined]
        file, model_name
    )

    aoti_compiled_model = AOTICompiledModel(
        torch._C._aoti.AOTIModelPackageLoader(
            file,
            model_name,
            run_single_threaded,
            num_runners,
            device_idx,
        )
    )

    device = loaded_metadata["AOTI_DEVICE_KEY"]
    from torch._inductor.codecache import get_device_information

    current_device_info = get_device_information(device)

    for k, v in current_device_info.items():
        if k in loaded_metadata:
            if v != loaded_metadata[k]:
                logger.warning(
                    "Device information mismatch for %s: %s vs %s. "
                    "This could cause some issues when loading the AOTInductor compiled artifacts.",
                    k,
                    v,
                    loaded_metadata[k],
                )

    return aoti_compiled_model


def load_pt2(
    f: FileLike,
    *,
    expected_opset_version: dict[str, int] | None = None,
    run_single_threaded: bool = False,
    num_runners: int = 1,
    device_index: int = -1,
    load_weights_from_disk: bool = False,
) -> PT2ArchiveContents:  # type: ignore[type-arg]
    """
    Loads all the artifacts previously saved with ``package_pt2``.

    Args:
        f (str | os.PathLike[str] | IO[bytes]): A file-like object (has to
         implement write and flush) or a string containing a file name.

        expected_opset_version (Optional[Dict[str, int]]): A map of opset names
         to expected opset versions

        num_runners (int): Number of runners to load AOTInductor artifacts

        run_single_threaded (bool): Whether the model should be run without
            thread synchronization logic. This is useful to avoid conflicts with
            CUDAGraphs.

        device_index (int): The index of the device to which the PT2 package is
            to be loaded. By default, `device_index=-1` is used, which corresponds
            to the device `cuda` when using CUDA. Passing `device_index=1` would
            load the package to `cuda:1`, for example.

    Returns:
        A ``PT2ArchiveContents`` object which contains all the objects in the PT2.
    """

    from torch._inductor.cpp_builder import normalize_path_separator

    if not (
        (isinstance(f, (io.IOBase, IO)) and f.readable() and f.seekable())
        or (isinstance(f, (str, os.PathLike)) and os.fspath(f).endswith(".pt2"))
    ):
        # TODO: turn this into an error in 2.9
        logger.warning(
            "Unable to load package. f must be a buffer or a file ending in "
            ".pt2. Instead got {%s}",
            f,
        )

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    weights = {}
    weight_maps = {}
    # pyrefly: ignore [bad-argument-type]
    with PT2ArchiveReader(f) as archive_reader:
        version = archive_reader.read_string(ARCHIVE_VERSION_PATH)
        if version != ARCHIVE_VERSION_VALUE:
            raise ValueError(
                f"Saved archive version {version} does not match our current "
                f"archive version {ARCHIVE_VERSION_VALUE}."
            )

        file_names = archive_reader.get_file_names()

        exported_programs = _load_exported_programs(
            archive_reader, file_names, expected_opset_version
        )
        extra_files = _load_extra_files(archive_reader, file_names)

        # Get a list of AOTI model names
        aoti_model_names: set[str] = set()
        for file in file_names:
            if file.startswith(AOTINDUCTOR_DIR):
                file_end = file[
                    len(AOTINDUCTOR_DIR) :
                ]  # remove data/aotinductor/ prefix
                file_end = normalize_path_separator(
                    file_end
                )  # Win32 need normalize path before split.
                model_name = file_end.split("/")[
                    0
                ]  # split "model_name/...cpp" into "model_name"
                aoti_model_names.add(model_name)
                if load_weights_from_disk and file.endswith("weights_config.json"):
                    weight_map = json.loads(archive_reader.read_string(file))
                    weight_maps[model_name] = weight_map
            elif load_weights_from_disk and file.startswith(WEIGHTS_DIR):
                weight_file_name = file[
                    len(WEIGHTS_DIR) :
                ]  # remove data/weights/ prefix
                weight_bytes = archive_reader.read_bytes(file)
                loaded_weight = torch.load(io.BytesIO(weight_bytes))
                weights[weight_file_name] = loaded_weight

    if isinstance(f, (io.IOBase, IO)):
        if len(aoti_model_names) > 0:
            # Workaround for AOTIModelPackageLoader not reading buffers
            with tempfile.NamedTemporaryFile(suffix=".pt2") as tf:
                f.seek(0)
                tf.write(f.read())
                f.seek(0)
                logger.debug("Writing buffer to tmp file located at %s.", tf.name)

                aoti_runners = {
                    model_name: _load_aoti(
                        tf.name,
                        model_name,
                        run_single_threaded,
                        num_runners,
                        device_index,
                    )
                    for model_name in aoti_model_names
                }
        else:
            aoti_runners = {}
    else:
        aoti_runners = {
            model_name: _load_aoti(
                f,
                model_name,
                run_single_threaded,
                num_runners,
                device_index,
            )
            for model_name in aoti_model_names
        }

    if weight_maps:
        for model_name in aoti_model_names:
            model_weights = {}
            for weight_name, (file, shape, stride, storage_offset) in weight_maps[
                model_name
            ].items():
                weight = weights[file]
                model_weights[weight_name] = weight.as_strided(
                    shape, stride, storage_offset
                )

            # user_managed=True ensures the weights updates are shared by all runners.
            aoti_runners[model_name].load_constants(
                model_weights, check_full_update=True, user_managed=True
            )

    return PT2ArchiveContents(exported_programs, aoti_runners, extra_files)


def load_weights_to_pt2_contents(
    pt2_contents: PT2ArchiveContents, weights_map: dict[str, Any]
) -> None:
    """
    Load weights into the models in PT2 archive contents

    Args:
        pt2_contents (PT2ArchiveContents): The contents of the PT2 archive.
    """
    for model_name, weights in weights_map.items():
        if model_name not in pt2_contents.aoti_runners:
            raise RuntimeError(f"Model {model_name} not found in PT2 archive contents.")
        pt2_contents.aoti_runners[model_name].load_constants(
            weights, check_full_update=True, user_managed=True
        )
<|endoftext|>import collections
import warnings

import torch
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.utils._ordered_set import OrderedSet


def _end_ptr(tensor: torch.Tensor) -> int:
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + tensor.element_size()
    else:
        stop = tensor.data_ptr()
    return stop


class TensorProperties:
    def __init__(self, tensor: torch.Tensor):
        self.is_fake = is_fake_tensor(tensor)
        self.is_contiguous = tensor.is_contiguous()
        self.storage_ptr = None
        self.storage_size = None
        self.start = None
        self.end = None

        if not self.is_fake:
            # only get the storage pointer for real tensors
            # pyrefly: ignore [bad-assignment]
            self.storage_ptr = tensor.untyped_storage().data_ptr()
            if self.is_contiguous:
                # only get storage size and start/end pointers for contiguous tensors
                # pyrefly: ignore [bad-assignment]
                self.storage_size = tensor.untyped_storage().nbytes()
                # pyrefly: ignore [bad-assignment]
                self.start = tensor.data_ptr()
                # pyrefly: ignore [bad-assignment]
                self.end = _end_ptr(tensor)

        # info to recover tensor
        self.shape = tensor.shape
        self.stride = tensor.stride()
        self.offset = tensor.storage_offset()

    def is_complete(self) -> bool:
        """
        Whether the tensor completely overlaps with its underlying storage
        """
        if self.is_fake:
            # Theoretically, fake tensors should not appear in weights
            # But we handle this corner case to make it always complete
            return True
        if not self.is_contiguous:
            return False

        if self.storage_ptr is None:
            raise AssertionError("storage_ptr cannot be None for complete check")
        if self.storage_size is None:
            raise AssertionError("storage_size cannot be None for complete check")
        if self.start is None:
            raise AssertionError("start cannot be None for complete check")
        if self.end is None:
            raise AssertionError("end cannot be None for complete check")
        return (
            self.start == self.storage_ptr
            and self.end == self.storage_ptr + self.storage_size
        )


class Weights(dict):
    """
    A dictionary mapping from weight name to a tuple of (tensor, TensorProperties).
    tensor represents the actual initial value of the weight.
    TensorProperties represents the properties of the weight that are needed to recover the weight.

    We use two separate entries because `tensor` could be a clone of the original weight tensor,
    so it doesn't have the same property as the original weight (such as underlying storage pointer).
    """

    def __init__(self, weight_dict: dict[str, tuple[torch.Tensor, TensorProperties]]):
        super().__init__(weight_dict)

    def get_weight(self, name: str) -> tuple[torch.Tensor, TensorProperties]:
        return self[name]

    def get_weight_properties(self, name: str) -> TensorProperties:
        return self[name][1]


def get_complete_tensor(
    group: OrderedSet[tuple[str, str]], models_weights: dict[str, Weights]
) -> torch.Tensor:
    """
    Given a group of (model_name, weight_name) pairs whose tensors share the same
    underlying storage, return the complete (maximal) tensor covering that storage
    region.

    This function handles two cases:

    1. If any tensor in the group is already marked as complete, return it directly.
    2. Otherwise, all tensors in the group are assumed to be slices of a larger,
        contiguous tensor backed by the same storage. In this case, reconstruct
        the complete tensor by taking the union of their storage ranges. This assumes
        all tensors in the group have the same dtype.

    Args:
        group: Set of (model_name, weight_name) tuples identifying tensors that share storage.
        models_weights: Dictionary mapping model names to their Weights objects.

    Returns:
        The complete tensor (either found directly or reconstructed from slices).

    Example:
        # Tensors a, b, c share storage:
        # a = full_tensor[0:5]   -> start=addr_0, end=addr_5
        # b = full_tensor[3:8]   -> start=addr_3, end=addr_8
        # c = full_tensor        -> complete tensor

        # Case 1: If c is in group -> return c
        # Case 2: If only a, b in group -> reconstruct from addr_0 to addr_8
    """

    if len(group) == 0:
        raise AssertionError("group cannot be empty")

    start_addr = None
    end_addr = None
    for model_name, weight_name in group:
        tensor_property = models_weights[model_name].get_weight_properties(weight_name)

        # Case 1: Found a complete tensor.
        if tensor_property.is_complete():
            return models_weights[model_name].get_weight(weight_name)[0]

        # Case 2: Track the widest boundary across all slices.
        if tensor_property.start is not None:
            start_addr = (
                tensor_property.start
                if start_addr is None
                else min(start_addr, tensor_property.start)
            )
        if tensor_property.end is not None:
            end_addr = (
                tensor_property.end
                if end_addr is None
                else max(end_addr, tensor_property.end)
            )

    # Case 2: Reconstruct complete tensor from slices.
    # Pick any tensor from the group as a reference (they all share the same storage).
    warnings.warn(
        "No complete tensor found in the group! Returning the first one. "
        "This may cause issues when your weights are not on CPU.",
        stacklevel=2,
    )

    model_name, weight_name = next(iter(group))
    reference_tensor = models_weights[model_name].get_weight(weight_name)[0]

    # If no boundary information available (e.g., FakeTensor), return reference tensor as is.
    if start_addr is None and end_addr is None:
        return reference_tensor

    # Validate that we have both boundaries.
    if start_addr is None or end_addr is None:
        raise AssertionError(
            f"Inconsistent boundary information: start={start_addr}, end={end_addr}. "
            "Unable to reconstruct complete tensor from group."
        )

    # Reconstruct a view over the full contiguous storage range.
    storage = reference_tensor.untyped_storage()
    total_size_bytes = end_addr - storage.data_ptr()
    element_size = reference_tensor.element_size()
    # It assumes all tensors in the group have the same dtype.
    total_size = total_size_bytes // element_size

    # Validate alignment: size must be multiples of element_size.
    if total_size_bytes % element_size != 0:
        raise AssertionError(
            f"Total size ({total_size_bytes} bytes) is not aligned with "
            f"element size ({element_size} bytes). Cannot reconstruct tensor safely. "
            f"Expected size to be a multiple of {element_size}."
        )

    # Reconstruct a tensor that spans the needed storage range, the metadata will be handled separately.
    return torch.tensor(
        [], device=reference_tensor.device, dtype=reference_tensor.dtype
    ).set_(
        storage,
        0,
        torch.Size([total_size]),
        (),
    )


def group_weights(all_weights: dict[str, Weights]) -> list[OrderedSet[tuple[str, str]]]:
    """
    Group weights that share the same underlying storage.

    Returns a list of sets, each set contains a tuple of (model_name, weight_name).
    """

    weights_dict: dict[tuple[int, torch.dtype], OrderedSet[tuple[str, str]]] = (
        collections.defaultdict(OrderedSet)
    )  # (storage_key, dtype) -> set(weight)

    for model_name, weights in all_weights.items():
        for weight_name, (tensor, properties) in weights.items():
            weights_dict[(properties.storage_ptr, tensor.dtype)].add(
                (model_name, weight_name)
            )

    return list(weights_dict.values())
<|endoftext|>import copy
import dataclasses
import functools
import os
import types
import typing
import typing_extensions
import zipfile
from pathlib import Path

import torch
from torch.export.experimental._utils import _get_main_cpp_file, _get_make_file
from torch.export.exported_program import _decompose_exported_program
from torch.utils._ordered_set import OrderedSet


_InputT = typing_extensions.ParamSpec("_InputT")
_RetT = typing.TypeVar("_RetT")


__all__ = []  # type: ignore[var-annotated]


def _copy_graph_module_and_signature(
    ep: torch.export.ExportedProgram,
) -> tuple[torch.fx.GraphModule, torch.export.graph_signature.ExportGraphSignature]:
    # copy.deepcopy lets the objects override __deepcopy__ methods with graph_copy() and node_copy(),
    # and this can break placeholder names in some particular cases.
    # For example, node copying will avoid Python keywords like 'input', suffixing and renaming to 'input_1'.
    # So we manually overwrite placeholder names by reading the old graph.
    gm = copy.deepcopy(ep.graph_module)
    new_graph_signature = copy.deepcopy(ep.graph_signature)

    # iterate over old/new graph modules
    for old_gm, new_gm in zip(ep.graph_module.modules(), gm.modules()):  # type: ignore[union-attr]
        old_phs = [node for node in old_gm.graph.nodes if node.op == "placeholder"]
        new_phs = [node for node in new_gm.graph.nodes if node.op == "placeholder"]
        # iterate over placeholders
        if len(old_phs) != len(new_phs):
            raise AssertionError(
                f"Number of old placeholders ({len(old_phs)}) does not match "
                f"new placeholders ({len(new_phs)})"
            )
        for old_node, new_node in zip(old_phs, new_phs):
            new_node.name = old_node.name

    return gm, new_graph_signature


def _remove_detach_pass(
    gm: torch.fx.GraphModule, sig: torch.export.graph_signature.ExportGraphSignature
) -> None:
    with gm._set_replace_hook(sig.get_replace_hook()):
        for node in list(reversed(gm.graph.nodes)):
            if node.op != "call_function":
                continue
            if (
                node.target is torch.ops.aten.detach.default
                and len(node.users) == 1
                and next(iter(node.users)).target is torch.ops.aten.detach.default
            ):
                next(iter(node.users)).replace_all_uses_with(node)

    gm.graph.eliminate_dead_code()
    gm.recompile()


def _export_forward_backward(
    ep: torch.export.ExportedProgram, joint_loss_index: int = 0
) -> torch.export.ExportedProgram:
    """
    WARNING: This API is highly unstable and will be subject to change in the future.
    """
    from torch._decomp import core_aten_decompositions

    ep = _decompose_exported_program(
        ep,
        cia_to_decomp={},
        python_decomp_table=core_aten_decompositions(),
        joint_loss_index=joint_loss_index,
        # For serialization purpose, we don't want to decompose custom triton ops.
        # If users would like to decompose custom triton ops, they could do it
        # with run_decompositions() API.
        decompose_custom_triton_ops=False,
    )
    gm, new_graph_signature = _copy_graph_module_and_signature(ep)
    _remove_detach_pass(gm, new_graph_signature)

    return ep._update(gm, new_graph_signature)


def _sticky_export(
    forward_func: typing.Callable[_InputT, _RetT],
    dynamic_shapes_callback: typing.Callable[
        _InputT, list[typing.Any] | dict[str, typing.Any] | tuple[typing.Any, ...]
    ]
    | None = None,
) -> typing.Callable[_InputT, _RetT]:
    """
    Lazily export the model on first forward call.
    Usage:
        model.forward = _sticky_export(model.forward, dynamic_shapes_callback=callback)
    """
    model = forward_func.__self__  # type: ignore[attr-defined]
    original_forward = forward_func.__func__  # type: ignore[attr-defined]

    @functools.wraps(forward_func)
    def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        # Unpatch forward to avoid recursion during export
        model.forward = types.MethodType(original_forward, model)

        dynamic_shapes_spec = None
        if dynamic_shapes_callback:
            dynamic_shapes_spec = dynamic_shapes_callback(*args, **kwargs)

        try:
            exported = torch.export.export(
                model,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes_spec,
            ).module()
            wrapper._exported_artifact = exported  # type: ignore[attr-defined]
        finally:
            # Restore the wrapper after export
            model.forward = wrapper

        return exported(*args, **kwargs)

    return wrapper


@dataclasses.dataclass
class _ExportMethod:
    overloads: dict[str, torch.export.ExportedProgram]
    fallbacks: list[torch.export.ExportedProgram]


class _ExportPackage:
    """
    An export package is a collection of torch.export()-ed PyTorch models consisting of
    a list of exported methods and their corresponding overloads. ExportPackage is introduced
    on top of torch.export() to support the following use cases:
        - Exporting a model with multiple methods if a model has multiple independent parts.
        - Exporting a function with multiple overloads based on tensor shapes or other metadata.

    ExportPackage is designed to contain multiple methods (associated with method names) and for
    each method, it can have multiple overloads (associated with overload names).

    Here is an example of the data structure for an ExportPackage:
    ```
    ExportPackage(
        methods={
            "decoder": ExportMethod(
                overloads={
                    "prefill": ExportedProgram(...),
                    "decode": ExportedProgram(...),
                },
                fallbacks=[],
            ),
            "encoder": ExportMethod(overloads={}, fallbacks=[ExportedProgram(...)]),
        },
    )
    ```

    To export a model into an ExportPackage, users can use the exporter API provided by ExportPackage.
    Exporter is a decorator that takes a callable and returns a wrapper. The wrapper will export the
    function into an ExportPackage, when it's invoked with some sample inputs (similar to how
    torch.compile() works). For more details, please refer to the document on .exporter() method.

    This design allows users to decouple the exported callables from the actual sample inputs which can
    be helpful for use cases where the exported callable is hidden behind helper functions or when sample
    inputs are hard to get.

    NOTE: This is an experimental API and anything can be changed in the future.

    Example usage:
    ```
        def fn(x):
            return x + 1

        def main(f, x):
            x += 1
            ret = f(x)
            return ret + 1

        package = ExportPackage()
        main(package.exporter(fn), torch.randn(3, 2))
    ```

    """

    def __init__(self) -> None:
        self.methods: dict[str, _ExportMethod] = {}

    def _exporter(
        self,
        method: str,
        fn: typing.Callable[_InputT, _RetT],
        *,
        fallback: str = "once",
    ) -> typing.Callable[_InputT, _RetT]:
        """
        A function/module decorator that sets up a callable to be exported later invoked.
        By default the exporter will only trigger torch.export for once and error on
        later invocations. To customize this behavior, users have the following two options:
          1. Call .define_overload() method on the returned wrapper to define an overload.
          2. Adjust the fallback policy using `fallback` argument.

        An "overload" is a named branch for an ExportMethod with a user defined precondition,
        typically based on input tensor shapes. It's up to a downstream backend implementation
        of ExportMethod to respect the precondition later in inference.

        define_overload() takes arguments like the following:
          - A name, for indexing purposes in a backend.
          - A callable (spec) that:
            - Has the same model input signature as the original model code.
            - Returns an optional dynamic shape spec.

        Exporter will only export an overload when the spec callable successfully returns
        a result without raising AssertionError.

        For example:
        ```
        package = ExportPackage()


        def prefill(x, xa, kv_cache):
            assert x.shape[1] == 3
            assert kv_cache == {}


        def decode(x, xa, kv_cache):
            assert x.shape[1] > 1
            assert len(kv_cache) > 0
            return {...}  # dynamic shape specs here


        exporter = (
            package.exporter(decoder)
            .define_overload("prefill", prefill)
            .define_overload("decode", decode)
        )
        ```

        A "fallback" is exported when no overload precondition matches a given set of sample
        inputs. Overloads should
        Fallbacks don't have names and are ordered in a list. It's up to a backend to decide
        which fallback is used among multiple ones.

        A reference backend implementation of ExportMethod may look like the following:
        ```
        def execute(method: ExportMethod, *args, **kwargs):
            for overload in method.overloads:
                if match_precondition(overload, *args, **kwargs):
                    return execute_overload(overload, *args, **kwargs)
            for fallback in method.fallbacks:
                if match_precondition(fallback, *args, **kwargs):
                    return execute_fallback(fallback, *args, **kwargs)
        ```

        Args:
            method(str): The method name for an exported part of PyTorch model. This
                         will be saved together with the exported/compiled artifacts
                         in any serialization format and can be used as the key to
                         index ExportPackage methods later.
            fn(callable): A PyTorch function/module to be exported.
            fallback(str): The fallback policy to decide when to call torch.export
              - "once" is the default policy. Under this policy a PyTorch program is assumed
                to be only called once later and an error will be raised for subsequent
                runs.
              - "error" means the ExportMethod will never have any fallbacks, meaning
                users should define all the possible overloads ahead of time.

        """

        fallbacks: list[torch.export.ExportedProgram] = []
        specs: dict[str, typing.Callable[_InputT, typing.Any]] = {}
        overloads: dict[str, torch.export.ExportedProgram] = {}
        self.methods[method] = _ExportMethod(fallbacks=fallbacks, overloads=overloads)

        @functools.wraps(fn)
        def _exporter_context(*args, **kwargs):  # type: ignore[no-untyped-def]
            import torch.export._wrapper_utils

            model: torch.nn.Module
            if not isinstance(fn, torch.nn.Module):
                model = torch.export._wrapper_utils._WrapperModule(fn)
            else:
                model = fn

            for k, v in specs.items():
                try:
                    if isinstance(fn, torch.nn.Module):
                        dynamic_shapes = v(fn, *args, **kwargs)  # type: ignore[arg-type]
                    else:
                        # pyrefly: ignore [invalid-param-spec]
                        dynamic_shapes = v(*args, **kwargs)
                except AssertionError:
                    continue
                if k not in overloads:
                    ep = torch.export.export(
                        model, args, kwargs, dynamic_shapes=dynamic_shapes
                    )
                    overloads[k] = ep
                ep = overloads[k]
                return ep.module()(*args, **kwargs)

            if fallback == "error":
                raise RuntimeError(
                    f"Exporter: Cannot export fallback {fn} when fallback policy is set to 'error',"
                    + "please specify an overload or adjust the fallback policy."
                )
            elif fallback == "once":
                if len(fallbacks) > 0:
                    raise RuntimeError(
                        f"Exporter: Cannot export {fn} more than once, "
                        + "please specify an overload or adjust the fallback policy."
                    )
            else:
                raise RuntimeError(f"Unknown fallback policy: {fallback}")
            ep = torch.export.export(model, args, kwargs)

            fallbacks.append(ep)
            return ep.module()(*args, **kwargs)

        if isinstance(fn, torch.nn.Module):
            _exporter_context = torch._dynamo.eval_frame.OptimizedModule(  # type: ignore[assignment]
                fn,
                lambda _: _exporter_context,  # type: ignore[arg-type]
            )

        def _define_overload(
            overload: str, spec: typing.Callable[_InputT, typing.Any]
        ) -> typing.Any:
            if overload in specs:
                raise AssertionError(f"Overload '{overload}' already exists in specs")
            if not callable(spec):
                raise AssertionError(f"spec must be callable, but got {type(spec)}")
            if not overload.isidentifier():
                raise AssertionError(
                    f"Overload '{overload}' is not a valid Python identifier"
                )
            specs[overload] = spec
            return _exporter_context

        if hasattr(fn, "_define_overload"):
            raise AssertionError("fn already has a '_define_overload' attribute")
        _exporter_context._define_overload = _define_overload  # type: ignore[attr-defined]

        # pyrefly: ignore [bad-return]
        return _exporter_context

    @property
    def _method_overloads(
        self,
    ) -> typing.Iterator[tuple[str, torch.export.ExportedProgram]]:
        for method, method_data in self.methods.items():
            for overload, ep in method_data.overloads.items():
                yield f"{method}:{overload}", ep

    def _compiled_and_package(
        self,
        f: torch.types.FileLike,
        standalone: bool = False,
        package_example_inputs: bool = False,
    ) -> None:
        options: dict[str, typing.Any] = {
            "aot_inductor.package": True,
            "aot_inductor.package_cpp_only": True,
            "always_keep_tensor_constants": True,
            # we'll change this back to False once we enable weight deduping for standalone mode
            "aot_inductor.package_constants_in_so": standalone,
            "aot_inductor_mode.compile_standalone": standalone,
        }
        aoti_files_map = {}
        model_names = []
        device_type = "cpu"
        for name, ep in self._method_overloads:
            name = name.replace(":", "__")
            model_names.append(name)
            options["aot_inductor.model_name_for_generated_files"] = name
            aoti_files = torch._inductor.aot_compile(
                ep.module(),  # type: ignore[arg-type]
                ep.example_inputs[0],
                kwargs=ep.example_inputs[1],
                options=options,
            )
            # pyrefly: ignore [unsupported-operation]
            aoti_files_map[name] = aoti_files

        from torch._inductor.package import package

        pt2_path = package.package_aoti(
            f,
            aoti_files_map,  # type: ignore[arg-type]
        )

        if not standalone:
            return

        if not isinstance(pt2_path, str):
            raise AssertionError(
                f"Expected pt2_path to be a string, but got {type(pt2_path)}"
            )
        base_directory = os.path.dirname(pt2_path)
        package_name = os.path.basename(pt2_path)[:-4]
        with zipfile.ZipFile(pt2_path, "r") as zip_ref:
            zip_ref.extractall(base_directory)

        example_inputs_map: dict[str, int] | None = (
            {} if package_example_inputs else None
        )
        for name, ep in self._method_overloads:
            name = name.replace(":", "__")
            # TODO: also dump kwargs
            # TODO: currently only support list of Tensors and they need to be on the same device
            if not ep.example_inputs:
                continue

            device_types: OrderedSet[str] = OrderedSet()

            for inp in ep.example_inputs[0]:
                if isinstance(inp, torch.Tensor):
                    device_types.add(inp.device.type)
            device_types.discard("cpu")
            if len(device_types) > 1:
                raise AssertionError(
                    "Does not support mixing {}".format("+".join(list(device_types)))
                )
            device_type = "cpu" if len(device_types) == 0 else device_types.pop()

            if package_example_inputs:
                if example_inputs_map is None:
                    raise AssertionError(
                        "example_inputs_map cannot be None when package_example_inputs is True"
                    )
                example_inputs_map[name] = len(ep.example_inputs[0])
                for i, t in enumerate(ep.example_inputs[0]):
                    path = Path(base_directory) / f"{name}_input_{i}.pt"
                    torch.save(t, path)

        cmake_file_str = _get_make_file(
            package_name, model_names, device_type=device_type
        )

        with open(Path(base_directory) / "CMakeLists.txt", "w") as file:
            file.write(cmake_file_str)

        main_file_str = _get_main_cpp_file(
            package_name, model_names, example_inputs_map, device_type=device_type
        )
        with open(Path(base_directory) / "main.cpp", "w") as file:
            file.write(main_file_str)
<|endoftext|>import logging

import torch
from torch._inductor.utils import IndentedBuffer


__all__ = []  # type: ignore[var-annotated]
logger = logging.getLogger(__name__)


def _get_main_cpp_file(
    package_name: str,
    model_names: list[str],
    example_inputs_map: dict[str, int] | None,
    device_type: str,
) -> str:
    """
    Generates a main.cpp file for AOTInductor standalone models in the specified package.

    Args:
        package_name (str): Name of the package containing the models.
        model_names (List[str]): List of model names to include in the generated main.cpp.
        cuda (bool): Whether to generate code with CUDA support.
        example_inputs_map (Optional[Dict[str, List[Tensor]]]): A mapping from model name to
            its list of example input tensors. If provided, the generated main.cpp will
            load and run these inputs.

    Returns:
        str: The contents of the generated main.cpp file as a string.
    """

    ib = IndentedBuffer()

    ib.writelines(
        [
            "#include <dlfcn.h>",
            "#include <fstream>",
            "#include <iostream>",
            "#include <memory>",
            "#include <torch/torch.h>",
            "#include <vector>",
            "#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>",
        ]
    )
    if device_type == "cuda":
        if torch.version.hip:
            ib.writelines(
                [
                    "#include <hip/hip_runtime.h>",
                ]
            )

        else:
            ib.writelines(
                [
                    "#include <cuda.h>",
                    "#include <cuda_runtime_api.h>",
                ]
            )
    for model_name in model_names:
        ib.writeline(
            f'#include "{package_name}/data/aotinductor/{model_name}/{model_name}.h"'
        )

    ib.newline()
    for model_name in model_names:
        ib.writeline(f"using torch::aot_inductor::AOTInductorModel{model_name};")

    ib.writelines(
        [
            "using torch::aot_inductor::ConstantHandle;",
            "using torch::aot_inductor::ConstantMap;",
            "",
            "int main(int argc, char* argv[]) {",
        ]
    )

    with ib.indent():
        ib.writeline(f'std::string device_str = "{device_type}";')
        ib.writeline("try {")

        with ib.indent():
            ib.writeline("c10::Device device(device_str);")

            if example_inputs_map is not None:
                # TODO: add device
                for i, model_name in enumerate(model_names):
                    num_inputs = example_inputs_map[model_name]

                    ib.writeline(f"// Load input tensors for model {model_name}")
                    ib.writeline(f"std::vector<at::Tensor> input_tensors{i + 1};")
                    ib.writeline(f"for (int j = 0; j < {num_inputs}; ++j) {{")
                    with ib.indent():
                        ib.writeline(
                            f'std::string filename = "{model_name}_input_" + std::to_string(j) + ".pt";'
                        )
                        ib.writeline("std::ifstream in(filename, std::ios::binary);")
                        ib.writeline("if (!in.is_open()) {")
                        with ib.indent():
                            ib.writeline(
                                'std::cerr << "Failed to open file: " << filename << std::endl;'
                            )
                            ib.writeline("return 1;")
                        ib.writeline("}")
                        ib.writeline(
                            "std::vector<char> buffer((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());"
                        )
                        ib.writeline(
                            "torch::IValue ivalue = torch::pickle_load(buffer);"
                        )
                        ib.writeline(
                            f"input_tensors{i + 1}.push_back(ivalue.toTensor().to(device));"
                        )
                    ib.writeline("}")
                    ib.newline()

                ib.newline()
                ib.writeline("\n// Create array of input handles")
                for i in range(len(model_names)):
                    ib.writelines(
                        [
                            f"auto input_handles{i + 1} =",
                            f"    torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(input_tensors{i + 1});",
                        ]
                    )

                ib.writeline("\n// Create array for output handles")
                for i in range(len(model_names)):
                    ib.writeline(f"AtenTensorHandle output_handle{i + 1};")

            ib.writeline("\n// Create and load models")
            for i, model_name in enumerate(model_names):
                ib.writelines(
                    [
                        f"auto constants_map{i + 1} = std::make_shared<ConstantMap>();",
                        f"auto constants_array{i + 1} = std::make_shared<std::vector<ConstantHandle>>();",
                        f"auto model{i + 1} = std::make_unique<AOTInductorModel{model_name}>(",
                        f"    std::move(constants_map{i + 1}),",
                        f"    std::move(constants_array{i + 1}),",
                        "    device_str,",
                        f'    "{package_name}/data/aotinductor/{model_name}/");',
                        f"model{i + 1}->load_constants();",
                    ]
                )

            if example_inputs_map is not None:
                ib.writeline("\n// Run the models")
                for i in range(len(model_names)):
                    ib.writeline(
                        f"torch::aot_inductor::DeviceStreamType stream{i + 1} = nullptr;"
                    )
                    ib.writeline(
                        f"model{i + 1}->run(&input_handles{i + 1}[0], &output_handle{i + 1}, stream{i + 1}, nullptr);"
                    )

                ib.writeline("\n// Convert output handles to tensors")
                for i in range(len(model_names)):
                    ib.writelines(
                        [
                            f"auto output_tensor{i + 1} =",
                            f"    torch::aot_inductor::alloc_tensors_by_stealing_from_handles(&output_handle{i + 1}, 1);",
                        ]
                    )

                ib.writeline("\n// Validate outputs")
                for i in range(len(model_names)):
                    ib.writeline(
                        f"""std::cout << "output_tensor{i + 1}\\n" << output_tensor{i + 1} << std::endl;"""
                    )
                    ib.writeline(
                        f"""torch::save(output_tensor{i + 1}, "output_tensor{i + 1}.pt");"""
                    )

            ib.writeline("return 0;")

        ib.writelines(
            [
                "} catch (const std::exception &e) {",
            ]
        )
        with ib.indent():
            ib.writeline('std::cerr << "Error: " << e.what() << std::endl;')
            ib.writeline("return 1;")

        ib.writeline("}")
    ib.writeline("}")

    return ib.getvalue()


def _get_make_file(package_name: str, model_names: list[str], device_type: str) -> str:
    ib = IndentedBuffer()

    ib.writelines(
        [
            "cmake_minimum_required(VERSION 3.10)",
            "project(TestProject)",
            "",
            "set(CMAKE_CXX_STANDARD 20)",
            "",
        ]
    )

    from torch._inductor.config import test_configs

    if test_configs.use_libtorch:
        ib.writeline("find_package(Torch REQUIRED)")

    if device_type == "cuda":
        if torch.version.hip:
            ib.writeline("find_package(hip REQUIRED)")
        else:
            ib.writeline("find_package(CUDA REQUIRED)")

    ib.newline()
    for model_name in model_names:
        ib.writeline(f"add_subdirectory({package_name}/data/aotinductor/{model_name}/)")

    ib.writeline("\nadd_executable(main main.cpp)")
    if device_type == "cuda":
        if torch.version.hip:
            ib.writeline("target_compile_definitions(main PRIVATE USE_HIP)")
        else:
            ib.writeline("target_compile_definitions(main PRIVATE USE_CUDA)")
    elif device_type == "xpu":
        ib.writeline("target_compile_definitions(main PRIVATE USE_XPU)")

    model_libs = " ".join(model_names)
    ib.writeline(f"target_link_libraries(main PRIVATE torch {model_libs})")

    if device_type == "cuda":
        if torch.version.hip:
            ib.writeline("target_link_libraries(main PRIVATE hip::host)")
        else:
            ib.writeline("target_link_libraries(main PRIVATE cuda ${CUDA_LIBRARIES})")
    elif device_type == "xpu":
        ib.writeline("target_link_libraries(main PRIVATE sycl ze_loader)")
    return ib.getvalue()
<|endoftext|># mypy: allow-untyped-defs
import inspect
import typing
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import lru_cache, partial, wraps
from itertools import chain
from typing import Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import ParamSpec


if TYPE_CHECKING:
    from torch.export.decomp_utils import CustomDecompTable

import torch
import torch.library
from torch._ops import HigherOrderOperator, OperatorBase, OpOverload, OpOverloadPacket
from torch._prims_common import CustomOutParamAnnotation
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.utils import _pytree as pytree


__all__ = [
    "decomposition_table",
    "pre_autograd_decomposition_table",
    "meta_table",
    "register_decomposition",
    "get_decompositions",
    "core_aten_decompositions",
    "_should_decompose_because_unsafe_op",
]

_T = TypeVar("_T")
_P = ParamSpec("_P")

# TODO: relax key type here; torch registrations should be possible to; but
# right now this type is accurate
global_decomposition_table: dict[str, dict[torch._ops.OperatorBase, Callable]] = (
    defaultdict(dict)
)

decomposition_table = global_decomposition_table["post_autograd"]
pre_autograd_decomposition_table = global_decomposition_table["pre_autograd"]
meta_table = global_decomposition_table["meta"]


def _should_decompose_because_unsafe_op(op: torch._ops.OperatorBase) -> bool:
    """
    Returns True if the op must always decompose in export/compile tracing system

    In export, we always decompose certain CIA ops that are tagged with
    maybe_aliasing_or_mutating because we statically need to know if the op is
    mutating or not. But these CIA ops could have different behaviour in runtime.

    native_batch_norm is a prim op which has a wrong schema and it needs to be replaced
    with correct schema. But until then, we will force decompose it via this tag.
    """
    if not isinstance(op, torch._ops.OpOverload):
        return False
    if torch.Tag.maybe_aliasing_or_mutating in op.tags:
        return True
    return op is torch.ops.aten.native_batch_norm.default


def _add_op_to_registry(registry, op, fn):
    """
    This is an internal API for adding an op to the decomposition table.

    If op is OpOverload, it will be added to the registry directly.
    If op is OpOverloadPacket, all the valid overload_ops in the packet will be added to the registry.
    """
    overloads: list[torch._ops.OpOverload] = []
    if isinstance(op, HigherOrderOperator):
        # There's no concept of overloads for HigherOrderOperator
        registry[op] = fn
        return
    elif isinstance(op, OpOverload):
        overloads.append(op)
    else:
        if not isinstance(op, OpOverloadPacket):
            raise AssertionError(f"expected OpOverloadPacket, got {type(op)}")
        overloads.extend(op.op_overloads())

    for op_overload in overloads:
        if op_overload in registry:
            raise RuntimeError(f"duplicate registrations for {op_overload}")
        # TorchScript dumps a bunch of extra nonsense overloads
        # which don't have corresponding dispatcher entries, we need
        # to filter those out, e.g aten.add.float_int
        if torch._C._dispatch_has_kernel(op_overload.name()):
            registry[op_overload] = fn
            schema = op_overload._schema
            if registry is decomposition_table:
                torch._C._fake_dispatch_register_decomp(
                    schema.name, schema.overload_name
                )
            elif registry is meta_table:
                torch._C._fake_dispatch_register_meta(schema.name, schema.overload_name)
            elif registry is pre_autograd_decomposition_table:
                # pre-autograd decomp not used by C++ faketensor
                pass
            elif any(registry is t for t in global_decomposition_table.values()):
                raise AssertionError(
                    f"decomposition registry {registry!r} is not mirrored into "
                    "the C++ fake dispatch tables"
                )


def _convert_out_params(f):
    out_annotation = f.__annotations__.get("out")

    # If there are no out params, do not wrap the function.
    if not out_annotation:
        return f

    # Hack to detect when out is a Tuple. There seems to be no pretty way of doing this
    if getattr(out_annotation, "__origin__", None) is tuple:
        sig = inspect.signature(f)
        out_names = sig.return_annotation._fields
        # If out is a tuple, we need to register a function that unpacks all the out
        # elements as this is what native_functions.yaml expects

        @wraps(f)
        def _fn(*args, **kwargs):
            out_kwargs = tuple(kwargs.pop(o, None) for o in out_names)
            # Either all of the out kwargs are set or none of them
            is_none = out_kwargs[0] is None
            if not all((o is None) == is_none for o in out_kwargs):
                raise AssertionError(
                    f"all out kwargs must be set or none of them, got {out_kwargs}"
                )
            return f(*args, **kwargs, out=None if is_none else out_kwargs)

        out_params = [
            inspect.Parameter(
                o,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=t,
            )
            for o, t in zip(out_names, typing.get_args(out_annotation))
        ]
        # Drop the out parameter and concatenate the new kwargs in the signature
        params = chain((v for k, v in sig.parameters.items() if k != "out"), out_params)
        _fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=params,  # type: ignore[arg-type]
            return_annotation=sig.return_annotation,
        )
        # Drop the out parameter and concatenate the new kwargs in the annotations
        _fn.__annotations__ = {k: v for k, v in f.__annotations__.items() if k != "out"}
        for o in out_params:
            _fn.__annotations__[o.name] = o.annotation

        # Propagate that this function is wrapped by `out_wrapper`
        _fn._torch_decompositions_out_wrapper = f._torch_decompositions_out_wrapper  # type: ignore[attr-defined]

        return _fn

    # Alternatively, there may be a single tensor out parameter with a name
    # other than "out". This will need special treatment and is indicated by an
    # annotation, which we will remove here so it is not exposed after wrapping.
    custom_out_param_name = f.__annotations__.pop(CustomOutParamAnnotation, None)
    if custom_out_param_name:

        @wraps(f)
        def _fn(*args, **kwargs):
            out_kwarg = kwargs.pop(custom_out_param_name, None)
            return f(*args, **kwargs, out=out_kwarg)

        out_param = inspect.Parameter(
            custom_out_param_name,
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=out_annotation,
        )

        # Drop the out parameter and concatenate the new kwarg in the signature
        sig = inspect.signature(f)
        params = chain(
            (v for k, v in sig.parameters.items() if k != "out"), (out_param,)
        )
        _fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=params,  # type: ignore[arg-type]
            return_annotation=sig.return_annotation,
        )

        # Drop the out parameter and concatenate the new kwargs in the annotations
        _fn.__annotations__ = {k: v for k, v in f.__annotations__.items() if k != "out"}
        _fn.__annotations__[out_param.name] = out_param.annotation

        return _fn

    return f


def register_decomposition(
    aten_op, registry=None, *, type="post_autograd", unsafe=False
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    A decorator to register a function as a decomposition to the Python
    decomposition table.  Use it like this::

        @register_decomposition(torch.ops.aten.clamp_min)
        def clamp_min(x):
            return torch.clamp(self, min=min)

    If you are writing a new decomposition, consider contributing it
    directly to PyTorch in torch._decomp.decompositions.

    This API is experimental; we are almost certainly going to extend
    the API when we make decompositions eligible for use in transforms (e.g.,
    autograd) and not just backend tracing, where we then need to know if a
    decomposition can be used to simulate a transform.

    By default, we also will register it to the Meta key of dispatcher,
    and replace the c++ Meta implementation if there is already one.

    unsafe kwarg is for reuse of this function for registering non-function
    things
    """

    if type not in {"post_autograd", "pre_autograd", "meta"}:
        raise AssertionError(
            f"type must be one of post_autograd, pre_autograd, or meta, got {type}"
        )

    def decomposition_decorator(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        orig_fn = fn
        if not unsafe:
            fn = _convert_out_params(fn)

        nonlocal registry
        if registry is None:
            registry = global_decomposition_table[type]

        def register(op):
            _add_op_to_registry(registry, op, fn)

        # To handle allowing multiple aten_ops at once
        pytree.tree_map_(register, aten_op)
        return orig_fn

    return decomposition_decorator


def get_decompositions(
    aten_ops: Sequence[torch._ops.OperatorBase | OpOverloadPacket],
    type: str = "post_autograd",
) -> dict[torch._ops.OperatorBase, Callable]:
    """
    Retrieve a dictionary of decompositions corresponding to the list of
    operator overloads and overload packets passed as input.  Overload
    packets will include all decomposed overloads in the packet.  If there is
    no decomposition for a requested operator, it is silently ignored.

    This API is experimental; we are almost certainly going to give an alternate,
    more recommended formulation, where a user provides the set of operators
    they know how to implement, and we provide decompositions for everything
    not in this set.
    """
    if type not in {"post_autograd", "pre_autograd", "meta"}:
        raise AssertionError(
            f"type must be one of post_autograd, pre_autograd, or meta, got {type}"
        )

    registry = global_decomposition_table[type]
    packets_to_overloads = defaultdict(list)

    for opo in registry:
        if isinstance(opo, (OpOverload, OpOverloadPacket)):
            packets_to_overloads[opo.overloadpacket].append(opo)
    decompositions: dict[torch._ops.OperatorBase, Callable] = {}
    for op in aten_ops:
        if isinstance(op, OpOverloadPacket) and op in packets_to_overloads:
            for op_overload in packets_to_overloads[op]:
                decompositions[op_overload] = registry[op_overload]
        elif isinstance(op, (torch._ops.OperatorBase)) and op in registry:
            decompositions[op] = registry[op]
    return decompositions


def remove_decompositions(
    decompositions: dict[torch._ops.OperatorBase, Callable],
    aten_ops: Sequence[OpOverload | OpOverloadPacket],
) -> None:
    """
    Given a dictionary of decompositions obtained from get_decompositions(), removes
    operators associated with a list of operator overloads and overload packets passed
    as input. If the decomposition dictionary does not contain a decomposition that is
    specified to be removed, it is silently ignored.
    """
    for op in aten_ops:
        if isinstance(op, OpOverloadPacket):
            for opo in op.op_overloads():
                decompositions.pop(opo, None)
        elif isinstance(op, OpOverload):
            decompositions.pop(op, None)


# populate the table
import torch._decomp.decompositions
import torch._refs


def core_aten_decompositions() -> "CustomDecompTable":
    from torch.export.exported_program import default_decompositions

    return default_decompositions()


# See NOTE [Core ATen Ops]
#
# list was copied from torch/_inductor/decomposition.py
# excluding decompositions that results in prim ops
# Resulting opset of decomposition is core aten ops
def _core_aten_decompositions_post_autograd() -> dict[
    torch._ops.OperatorBase, Callable
]:
    aten = torch.ops.aten
    return get_decompositions(
        [
            aten.addcdiv,
            aten.addcdiv_,
            aten.addcmul,
            aten.addcmul_,
            aten.addr,
            aten.affine_grid_generator,
            aten.alias_copy,
            aten.all,
            aten.aminmax,
            aten.arange.default,
            aten.arange.start,
            aten.avg_pool2d_backward,
            aten.baddbmm,
            aten.binary_cross_entropy,
            aten.binary_cross_entropy_backward,
            aten.binary_cross_entropy_with_logits,
            aten.block_diag,
            aten.bernoulli.p,
            aten.bernoulli.default,
            aten.celu,
            aten.celu_,
            aten.channel_shuffle,
            aten.clamp_max,
            aten.clamp_min,
            aten.col2im,
            aten.count_nonzero,
            aten.linalg_cross,
            aten.cudnn_batch_norm,
            aten.cudnn_batch_norm_backward,
            aten.miopen_batch_norm_backward,
            aten.deg2rad,
            aten.deg2rad_,
            aten.detach,
            aten.diag_embed,
            aten.diagonal_backward,
            aten.diagonal_copy,
            aten.dot,
            aten.vdot,
            aten.elu_,
            aten.elu_backward,
            aten._embedding_bag,
            aten.embedding_dense_backward,
            aten.empty_like,
            aten._euclidean_dist.default,
            aten.expand_as,
            aten.expand_copy,
            aten.eye,
            aten.fill,
            aten.fill_,
            aten.floor_divide,
            aten.frac,
            aten.frac_,
            aten._fused_moving_avg_obs_fq_helper,
            aten.gelu_,
            aten.gelu_backward,
            aten.glu,
            aten.glu_backward,
            aten.hardshrink,
            aten.hardsigmoid,
            aten.hardsigmoid_,
            aten.hardsigmoid_backward,
            aten.hardswish,
            aten.hardswish_,
            aten.hardswish_backward,
            aten.hardtanh_,
            aten.hardtanh_backward,
            aten.hann_window,
            aten.heaviside,
            aten.heaviside_,
            aten.huber_loss,
            aten.huber_loss_backward,
            aten.im2col,
            aten.index_add.out,
            aten.index_add.default,
            aten.index_add_,
            aten.index_copy.out,
            aten.index_copy.default,
            aten.index_copy_,
            aten.index_fill.int_Scalar,
            aten.index_fill.int_Tensor,
            aten.index_fill.int_Scalar_out,
            aten.index_fill.int_Tensor_out,
            aten.index_fill_,
            aten.isin,
            aten.isneginf,
            aten.isposinf,
            aten.l1_loss,
            aten._lazy_clone,
            aten._test_parallel_materialize,
            aten.leaky_relu_,
            aten.leaky_relu_backward,
            aten.lerp,
            aten.lerp_,
            aten.linalg_vector_norm,
            aten.linspace,
            aten.logaddexp,
            aten.logaddexp2,
            aten.logit,
            aten.logit_,
            aten.logit_backward,
            aten.log_sigmoid_backward,
            aten.log_sigmoid_forward,
            aten._log_softmax_backward_data,
            aten.logspace,
            aten.logsumexp.default,
            aten.masked_fill,
            aten.masked_fill_,
            aten.max_unpool2d,
            aten.max_unpool3d,
            aten.mish,
            aten.mish_,
            aten.mish_backward,
            aten.mse_loss,
            aten.mse_loss_backward,
            aten.multi_margin_loss,
            aten.multilabel_margin_loss_forward,
            aten.mv,
            aten.mvlgamma,
            aten.mvlgamma_,
            aten.nansum,
            aten.nan_to_num,
            aten.nan_to_num_,
            aten.narrow,
            aten.native_batch_norm_backward,
            aten.native_dropout_backward,
            aten.native_group_norm_backward,
            aten.native_layer_norm_backward,
            aten._fused_rms_norm,
            aten._fused_rms_norm_backward,
            aten.new_empty,
            aten.new_full,
            aten.new_ones,
            aten.new_zeros,
            aten.nll_loss2d_forward,
            aten.nll_loss2d_backward,
            aten.nll_loss_backward,
            aten.nll_loss_forward,
            aten.norm.ScalarOpt_dtype,
            aten.norm.Scalar,
            aten.norm.ScalarOpt_dim_dtype,
            aten.norm.ScalarOpt_dim,
            aten.norm.dtype_out,
            aten.norm.out,
            aten.norm.ScalarOpt_dtype_out,
            aten.norm.Scalar_out,
            aten.ones,
            aten.ones_like,
            aten.pixel_shuffle,
            aten.pixel_unshuffle,
            aten._prelu_kernel,
            aten._prelu_kernel_backward,
            aten._reshape_alias,
            aten.rad2deg,
            aten.rad2deg_,
            aten.reflection_pad1d,
            aten.reflection_pad1d_backward,
            aten.reflection_pad2d,
            aten.reflection_pad2d_backward,
            aten.reflection_pad3d,
            aten.reflection_pad3d_backward,
            aten.replication_pad1d,
            aten.replication_pad2d,
            aten.replication_pad3d,
            aten.renorm,
            aten.renorm_,
            aten.replication_pad2d,
            aten.resize_as,
            aten.roll,
            aten.rot90,
            aten.rrelu_with_noise,
            aten.rrelu_with_noise_,
            aten.rsub,
            aten._safe_softmax,
            aten._scaled_dot_product_flash_attention_for_cpu.default,
            aten.select_backward,
            aten.select_scatter,
            aten.sgn,
            aten.sgn_,
            aten.sigmoid_backward,
            aten.silu,
            aten.silu_,
            aten.silu_backward.grad_input,
            aten.silu_backward,
            aten.sinc,
            aten.sinc_,
            aten.slice_backward,
            aten.smooth_l1_loss,
            aten.smooth_l1_loss_backward,
            aten.soft_margin_loss,
            aten.soft_margin_loss_backward,
            aten._softmax_backward_data,
            aten.softplus,
            aten.softplus_backward,
            aten.softshrink,
            aten.special_entr,
            aten.special_log_ndtr,
            aten.special_xlog1py,
            aten.split.Tensor,
            aten.split_with_sizes_copy,
            aten.squeeze_copy,
            aten.squeeze.default,
            aten.squeeze.dim,
            aten.std.correction,
            aten.std.out,
            aten.std.correction_out,
            aten.std_mean.correction,
            aten.std_mean.correction_out,
            aten.stack,
            aten.sum.default,
            aten.sum.out,
            aten.t,
            aten.t_copy,
            aten.take,
            aten.tanh_backward,
            aten.threshold,
            aten.threshold_,
            aten.threshold_backward,
            aten.trace,
            aten.transpose.int,
            aten.transpose_copy,
            aten.tril,
            aten.tril_,
            aten.triu,
            aten.triu_,
            aten.unbind,
            aten.unfold_backward,
            aten.unfold_copy,
            aten._unsafe_index,
            aten._unsafe_index_put,
            aten._unsafe_masked_index,
            aten._unsafe_masked_index_put_accumulate,
            aten.unsafe_split.Tensor,
            aten.unsafe_split_with_sizes,
            aten.unsqueeze_copy,
            aten._unsafe_view,
            aten.upsample_linear1d,
            aten.upsample_bilinear2d.out,
            aten.upsample_trilinear3d.out,
            aten.upsample_nearest2d_backward,
            aten.view_as_complex,
            aten.xlogy,
            aten.xlogy_,
            aten.zero,
            aten.zero_,
            aten.zeros,
            aten.zeros_like,
            aten._chunk_cat,
            aten._weight_norm_interface,
        ]
    )
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import inspect
from collections.abc import Callable

import torch
import torch._decomp
from torch import Tensor
from torch._prims_common.wrappers import _maybe_remove_out_wrapper


decomposition_table = torch._decomp.decomposition_table
decomposition_table_for_jvp: dict[torch._ops.OperatorBase, Callable] = {}
register_decomposition = torch._decomp.register_decomposition
aten = torch.ops.aten

# NOTE: [forward-mode AD decompositions mechanism]
#
# The mechanism is in VariableType,
#   IF any inputs have forward grad
#      AND there is no forward AD formula implemented
#      AND the functions are actually differentiable
#   run the decomposition
#      See run_jit_decomposition_with_args_for_jvp
#      We currently use python decompositions that we torchscript.
#
# Note that we would be building the backward graph at the decomposed level
# too, but that is OK, because we would've errored out otherwise anyway.
#
# TODO: The mechanism we are using to register decompositions doesn't
# seem to be exclusively used for jvp. So open question here is whether
# torch/csrc/jit/runtime/decomposition_registry.cpp is being used for other things.
# If that is the case, we may go down the decomposition path unexpectedly
# (and possibly produce an unintelligible error) vs erroring out earlier and
# printing that the forward AD formula is not implemented.
#
# The solution to this may be to have an explicit whitelist to control when
# to enable the decomposition.


def maybe_register_decomposition(op):
    def decorator(f):
        try:
            return register_decomposition(op)(f)
        except Exception:
            return f

    return decorator


# Functions where we need a special decomposition for jvp but there's another version that
# should be used more generally (ex. for jvp we need to recompute the mean and variance for
# the backwards of a normalization function. Without jvp, it should use the saved value)
decomposition_table_for_jvp = {}


def register_decomposition_for_jvp(fn):
    return register_decomposition(fn, registry=decomposition_table_for_jvp)


def _register_jit_decomposition_for_jvp(decomp, use_python=False):
    if decomp in decomposition_table_for_jvp:
        decomposition_table_used = decomposition_table_for_jvp
    elif decomp in decomposition_table:
        decomposition_table_used = decomposition_table
    else:
        raise RuntimeError(f"could not find decomposition for {decomp}")
    decomp_fn = decomposition_table_used[decomp]

    # `out_wrapper` extends a decompositions signature with
    # an `out` parameter. However jit will use the unwrapped function's
    # signature instead so we need to unwrap here to prevent an error
    decomp_fn = _maybe_remove_out_wrapper(decomp_fn)

    if use_python:
        decomp_fn = torch.jit.ignore(decomp_fn)
        sig = inspect.signature(decomp_fn)

        # Create a string wrapping the function from the signature
        # example output:
        # def wrapped_decomp(x: torch.Tensor, y: int, z: int):
        #   return decomp_fn(x, y, z)
        # Thanks copilot!
        def get_function_def(sig):
            param_def = [f"{param_str}" for param_str in sig.parameters.values()]
            param_use = [f"{param_str}" for param_str in sig.parameters]

            return f"def wrapped_decomp({', '.join(param_def)}):\n  return decomp_fn({', '.join(param_use)})\n"

        f_str = get_function_def(sig)
        graph = torch.jit.CompilationUnit(f_str).wrapped_decomp.graph
    else:
        graph = torch.jit.script(decomp_fn).graph
    torch.jit._register_decomposition(decomp, graph)


# The only decompositions here are temporary or hacks for the purposes of jvp


# TODO: do these also belong here?
@maybe_register_decomposition(aten.trace.default)
def trace(self: Tensor) -> Tensor:
    return torch.sum(torch.diag(self))


@maybe_register_decomposition(aten.log_sigmoid_forward.default)
def log_sigmoid_forward(self: Tensor) -> tuple[Tensor, Tensor]:
    min = torch.minimum(self.new_zeros(()), self)
    z = torch.exp(-torch.abs(self))
    if self.is_cuda or self.is_xpu:
        buffer = self.new_zeros((0,))
    else:
        buffer = z
    return min - torch.log1p(z), buffer


def recompute_mean_var(
    input: Tensor, rstd: Tensor, inner_dim_indices: list[int], keepdim: bool
):
    # for most norm decompositions, it will be the same as the core version except for here.
    # We recompute the mean and variance so that they track gradients through input

    var, mean = torch.var_mean(
        input, dim=inner_dim_indices, correction=0, keepdim=keepdim
    )
    eps = torch.pow(1 / rstd, 2) - var  # this makes me so sad inside
    eps = eps.detach()
    rstd = torch.rsqrt(var + eps)
    return mean, rstd


@register_decomposition_for_jvp(aten.native_layer_norm_backward)
def native_layer_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: list[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    output_mask: list[bool],
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    input_shape = input.shape
    input_ndim = input.dim()

    axis = input_ndim - len(normalized_shape)
    inner_dims = input_shape[axis:]
    outer_dims = input_shape[:axis]
    inner_dim_indices = list(range(axis, input_ndim))
    outer_dim_indices = list(range(axis))

    N = 1
    for i in inner_dims:
        N *= i
    M = 1
    for i in outer_dims:
        M *= i
    if M <= 0 or N <= 0:
        return (
            input.new_zeros(input_shape),
            input.new_zeros(input_shape[axis:]),
            input.new_zeros(input_shape[axis:]),
        )

    mean_, rstd_ = recompute_mean_var(input, rstd, inner_dim_indices, keepdim=True)

    x_hat = (input - mean_) * rstd_
    if weight is not None:
        grad_x_hat = grad_out * weight
    else:
        grad_x_hat = grad_out
    a = grad_x_hat * N
    b = torch.sum(grad_x_hat, inner_dim_indices, True)
    c1 = torch.mul(grad_x_hat, x_hat)
    c2 = torch.sum(c1, inner_dim_indices, True)
    c3 = torch.mul(x_hat, c2)
    inner = a - b - c3

    if output_mask[0]:
        d_input: Tensor | None = (rstd_ / N) * inner
    else:
        d_input = torch.zeros_like(input)  # should be None but doesn't work with vjp

    if output_mask[1] and weight is not None:
        if len(outer_dim_indices) > 0:
            d_weight: Tensor | None = torch.sum(
                grad_out * x_hat, outer_dim_indices, False
            )
        else:
            d_weight = grad_out * x_hat
    elif weight is not None:
        d_weight = torch.zeros_like(weight)  # should be None but doesn't work with vjp
    else:
        d_weight = torch.zeros(())  # should be None but doesn't work with vjp

    if output_mask[2] and bias is not None:
        if len(outer_dim_indices) > 0:
            d_bias: Tensor | None = torch.sum(grad_out, outer_dim_indices, False)
        else:
            d_bias = grad_out.clone()
    elif bias is not None:
        d_bias = torch.zeros_like(bias)  # should be None but doesn't work with vjp
    else:
        d_bias = torch.zeros(())  # should be None but doesn't work with vjp

    return (d_input, d_weight, d_bias)


def prod(x: list[int]):
    r = 1
    for i in x:
        r *= i
    return r


@register_decomposition_for_jvp(aten.native_batch_norm_backward)
def native_batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    save_mean: Tensor | None,
    save_invstd: Tensor | None,
    train: bool,
    eps: float,
    output_mask: list[bool],
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    input_shape = input.shape
    input_rank = input.dim()
    if input_rank < 2:
        raise AssertionError(f"rank of the input must be at least 2, got {input_rank}")

    axis = 1
    num_features = prod(input_shape) / input_shape[axis]  # type: ignore[arg-type]
    mean = save_mean
    invstd = save_invstd
    if train:
        if save_mean is None or save_invstd is None:
            raise AssertionError(
                "when train=True, save_mean and save_invstd are required"
            )

        reduciton_dims = [0] + list(range(2, input.dim()))
        if invstd is None:
            raise AssertionError("invstd must not be None for typing")
        mean, invstd = recompute_mean_var(input, invstd, reduciton_dims, keepdim=False)
    else:
        if running_mean is None or running_var is None:
            raise AssertionError(
                "running_mean and running_var must not be None when train=False"
            )
        mean = running_mean
        invstd = torch.rsqrt(running_var + eps)

    if invstd is None or mean is None:
        raise AssertionError(
            f"invstd and mean must not be None, got invstd={invstd}, mean={mean}"
        )

    broadcast_mask = [1] * input_rank
    broadcast_mask[axis] = input_shape[axis]

    reduction_axes: list[int] = []
    for i in range(input_rank):
        if i != axis:
            reduction_axes.append(i)

    mean = torch.reshape(mean, broadcast_mask)
    norm = 1.0 / num_features
    grad_output_sum = torch.sum(grad_out, reduction_axes)
    dot_p = torch.sum(grad_out * (input - mean), reduction_axes)

    grad_mean = torch.reshape(grad_output_sum * norm, broadcast_mask)
    proj_scale = torch.reshape(torch.mul(dot_p * norm, invstd * invstd), broadcast_mask)

    if weight is None:
        grad_scale = torch.reshape(invstd, broadcast_mask) * 1.0
    else:
        grad_scale = torch.reshape(invstd * weight, broadcast_mask)

    if train:
        proj = (input - mean) * proj_scale
        grad_input = ((grad_out - proj) - grad_mean) * grad_scale
    else:
        grad_input = grad_out * grad_scale

    if output_mask[1]:
        grad_weight = dot_p * invstd
    elif weight is not None:
        grad_weight = torch.zeros_like(
            weight
        )  # should be None but doesn't work with vjp
    else:
        grad_weight = torch.zeros(())  # should be None but doesn't work with vjp

    if output_mask[2]:
        grad_bias = grad_output_sum
    else:
        grad_bias = torch.zeros_like(
            grad_output_sum
        )  # should be None but doesn't work with vjp

    return (grad_input, grad_weight, grad_bias)


@register_decomposition_for_jvp(aten.batch_norm_backward)
def batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    save_mean: Tensor | None,
    save_var: Tensor | None,
    update: bool,
    eps: float,
    output_mask: list[bool],
    reserve: Tensor,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    return native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        update,
        eps,
        output_mask,
    )


_register_jit_decomposition_for_jvp(torch.ops.aten.trace.default, use_python=True)
_register_jit_decomposition_for_jvp(torch.ops.aten.nll_loss_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.nll_loss2d_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten._log_softmax_backward_data.default)
_register_jit_decomposition_for_jvp(torch.ops.aten._softmax_backward_data.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.log_sigmoid_forward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.native_layer_norm_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.native_batch_norm_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.cudnn_batch_norm_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.batch_norm_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.miopen_batch_norm_backward.default)
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
from collections import defaultdict
from collections.abc import Callable

import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload


aten = torch.ops.aten

rng_decompositions: dict[str, dict[OpOverload, Callable]] = defaultdict(dict)


def register_rng_decomposition(aten_op):
    return decomp.register_decomposition(aten_op, rng_decompositions)


def throw_on_non_cuda(device):
    raise RuntimeError(
        f"You are trying to functionalize a {device.type} RNG operator but {device.type} does not "
        f"use Philox/counter-based RNG. Therefore, functionalizing a {device.type} RNG operator is "
        "not supported. We are discussing the possibility of a Philox-based RNG implementation for CPU."
    )


# TODO - We have to register many more distributions here, and also higher level
# ops like dropout which have fused implementation and can hide the rand inside.
@register_rng_decomposition(aten.rand)
def rand(shape, dtype=None, layout=torch.strided, device=None, pin_memory=False):
    if device and device.type != "cuda":
        throw_on_non_cuda(device)
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    dtype = dtype or torch.float32
    out, offset_jump = torch.ops.rngprims.philox_rand(
        shape, seed, offset, None, device, dtype
    )
    PhiloxStateTracker.advance_offset(offset_jump)
    return out


@register_rng_decomposition(aten.rand_like)
def rand_like(
    x: torch.Tensor,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=torch.preserve_format,
):
    device = device or x.device
    if device.type != "cuda":
        throw_on_non_cuda(device)
    dtype = dtype or x.dtype
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    out, offset_jump = torch.ops.rngprims.philox_rand(
        x.shape, seed, offset, None, device, dtype
    )
    PhiloxStateTracker.advance_offset(offset_jump)
    return out


class PhiloxState:
    """
    Represents a PhiloxRngState - (seed, offset) where offset = base_offset +
    relative_offset. seed and base_offset basically point to the rng state just
    before tracing starts. relative offset tracks the totally consumed offset at
    trace time.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.seed = torch.tensor(())
        self.base_offset = torch.tensor(())
        self.relative_offset = 0
        self.offset_advanced_at_least_once = False

    def validate_state(self):
        if self.seed.numel() == 0 or self.base_offset.numel() == 0:
            raise AssertionError(
                f"seed and base_offset must not be empty, got "
                f"seed.numel()={self.seed.numel()}, base_offset.numel()={self.base_offset.numel()}"
            )

    def advance_offset(self, consumed_offset):
        self.offset_advanced_at_least_once = True
        self.relative_offset = self.relative_offset + consumed_offset

    def set_state(self, seed, base_offset, relative_offset=0):
        self.seed = seed
        self.base_offset = base_offset
        self.relative_offset = relative_offset

    def get_state_as_tuple(self):
        self.validate_state()
        return (self.seed, self.base_offset + self.relative_offset)

    def get_state_as_tensor(self):
        # Only needed because we override get_rng_state.
        self.validate_state()
        return torch.stack([self.seed, self.base_offset + self.relative_offset])

    def set_state_from_tensor(self, state):
        # Only needed because we override set_rng_state.
        self.seed, self.base_offset = torch.unbind(state)
        self.relative_offset = 0


class PhiloxStateTracker:
    """
    Singleton class to track the philox rng state during AOT Autograd tracing.
    For each aot tracing instance, AOT Autograd resets this tracker and keeps
    track of both forward and backward offsets. At runtime, we only care about
    the total consumed forward and backward offsets. For dynamic shapes, these
    offsets are a function of input shapes. Therefore, the AOT generated graphs
    have additional outputs that compute total consumed forward and backward
    offsets.
    """

    running_state: PhiloxState
    fwd_state: PhiloxState
    bwd_state: PhiloxState

    def __enter__(self):
        PhiloxStateTracker.reset()
        return self

    def __exit__(self, exc_type, exc_cal, exc_tb):
        PhiloxStateTracker.reset()

    @classmethod
    def reset(cls):
        cls.running_state = PhiloxState()
        cls.fwd_state = PhiloxState()
        cls.bwd_state = PhiloxState()

    @classmethod
    def mark_beginning_of_forward(cls):
        # Tells the tracker to use fwd_state as the running state
        cls.running_state = cls.fwd_state

    @classmethod
    def mark_beginning_of_backward(cls):
        # Tells the tracker to use bwd_state as the running state
        cls.running_state = cls.bwd_state

    @classmethod
    def record_state(cls, seed, offset, mode):
        # Records the seed and offset tensors. These tensors are used to invoke
        # the philox_rand functional primitives.
        if mode == "forward":
            cls.fwd_state.set_state(seed, offset)
            cls.mark_beginning_of_forward()
        else:
            if mode != "backward":
                raise AssertionError(f"mode must be 'backward', got {mode}")
            cls.bwd_state.set_state(seed, offset)

    @classmethod
    def get_state_as_tensor(cls):
        # The only reason this exists is because we override get_rng_state and
        # set_rng_state during tracing. get_rng_state expects a tensor output,
        # so return (seed, offset) tuple upset other parts of the program like
        # ctx.saved_tensors.

        # A bad consequence is that if user saves and restores rng state, we
        # have little bit of ugliness in the generated code, where we first
        # concat the (seed, offset) to create a tensor for get_rng_state, and
        # then split it back to get (seed, offset) tuple in set_rng_state.

        # TODO: Investigate if there is a better way to wrap the tuple in a
        # false Tensor object, and then desugar it later on.
        return cls.running_state.get_state_as_tensor()

    @classmethod
    def get_state_as_tuple(cls):
        return cls.running_state.get_state_as_tuple()

    @classmethod
    def set_state_from_tensor(cls, x):
        # This is only needed because we override set_rng_state. Look at the
        # comment in get_state_from_tensor method.
        cls.running_state.set_state_from_tensor(x)

    @classmethod
    def advance_offset(cls, consumed_offset):
        cls.running_state.advance_offset(consumed_offset)

    @classmethod
    def get_current_relative_offset(cls):
        return cls.running_state.relative_offset

    @staticmethod
    def multiple_of_4(offset):
        # torch cuda rng state offset must be a multiple of 4. For inductor, as
        # we sum up all the numel, the result might not be a multiple of 4. This
        # method achieves that.
        return (offset + 3) // 4 * 4

    @classmethod
    def get_updated_fwd_offset(cls):
        # Short circuit if no rand ops were observed
        if not cls.fwd_state.offset_advanced_at_least_once:
            return cls.fwd_state.base_offset
        return cls.multiple_of_4(
            cls.fwd_state.base_offset + cls.fwd_state.relative_offset
        )

    @classmethod
    def get_updated_bwd_offset(cls):
        # Short circuit if no rand ops were observed
        if not cls.bwd_state.offset_advanced_at_least_once:
            return cls.bwd_state.base_offset
        return cls.multiple_of_4(
            cls.bwd_state.base_offset + cls.bwd_state.relative_offset
        )


# Adding more decompositions which eventually use rand_like inside decomps.
# Adding these in rng_decompositions ensures the functionalization of rand_like
# ops used in these decomps. The list is copied from inductor codebase, which
# uses it for similar purpose.
#
# Caution - These decomps do not have same accuracy as that of eager. However,
# we can't just disable them with a config flag like fallback_random, because
# for functionalization of rng ops, we have to decompose these ops.
extra_random_decomps = get_decompositions(
    [
        aten.cauchy,
        aten.cauchy_,
        aten.exponential,
        aten.exponential_,
        aten.geometric,
        aten.geometric_,
        aten.native_dropout,
        aten.normal,
        aten.normal_,
        aten.normal_functional,
        aten.log_normal,
        aten.log_normal_,
        aten.rrelu_with_noise,
        aten.rrelu_with_noise_,
        aten.uniform,
        aten.uniform_,
    ]
)
register_extra_random_decomp = functools.partial(
    decomp.register_decomposition, registry=extra_random_decomps
)


@register_extra_random_decomp([aten.bernoulli_])
def bernoulli_(self, p=0.5):
    if self.device == torch.device("cpu"):
        return NotImplemented
    return self.copy_(torch.rand_like(self, dtype=torch.float32) < p)


@register_extra_random_decomp([aten.bernoulli.p])
def bernoulli_p(self, p=0.5, *, generator=None):
    if self.device == torch.device("cpu"):
        return NotImplemented
    if generator is not None:
        raise AssertionError(f"generator must be None, got {generator}")
    return torch.rand_like(self, dtype=torch.float32) < p


rng_decompositions.update(extra_random_decomps)  # type: ignore[arg-type]
<|endoftext|>from __future__ import annotations

from typing import Generic, TypeVar

import torch

# pyrefly: ignore [bad-dunder-all]
__all__ = ['Await']

W = TypeVar("W")

class _PyAwaitMeta(type(torch._C._Await), type(Generic)):  # type: ignore[misc, no-redef]
    pass

class _Await(torch._C._Await, Generic[W], metaclass=_PyAwaitMeta):
    r"""
    Wrapper around a ``torch._C.Await`` which encapsulates delayed execution
    of a callable. All manipulations happen with functions ``torch.jit._awaitable``,
    ``torch.jit._awaitable_wait``, ``torch.jit._awaitable_nowait``.

    Torch scriptable manipulations:
    ``torch.jit._awaitable(func, *args)``
    Creates ``Await[W]`` object, where W is return type of func.

    Returns:
    ``torch.jit._awaitable_wait(Await[W])``
    Returns the result of the function, specified at ``_awaitable``,  with specified arguments.

    Returns:
        The result of type ``W`` of the function call. The result is owned by ``Await[W]``
        and returned on all following ``_awaitable_wait`` calls.


    ``torch.jit._awaitable_nowait(W)``
    Returns:
        Trivial ``Await[W]`` with specified result.


    Only in eager mode:
    ``fn() -> Callable[Tuple[Any], W]``
    Returns:
        Specified at ``_awaitable`` python function ``func``.

    ``args() -> Tuple[Any]``
    Returns:
        Specified at ``_awaitable`` python args.

    ``is_nowait() -> _bool``
    Returns:
        ``True`` if this object was created via ``_awaitable_nowait`` call (trivial `Await[W]`).

    In eager mode ``Await[W]`` can be used as ``W`` i.e. attributes of W can be called on ``Await[W]``,
    ``_awaitable_wait()`` call will be transparently added.
    """
<|endoftext|># mypy: allow-untyped-defs
import sys
import types
from contextlib import contextmanager

import torch


# The idea for this parameter is that we forbid bare assignment
# to torch.backends.<cudnn|mkldnn>.enabled and friends when running our
# test suite, where it's very easy to forget to undo the change
# later.
__allow_nonbracketed_mutation_flag = True


def disable_global_flags():
    global __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = False


def flags_frozen():
    return not __allow_nonbracketed_mutation_flag


@contextmanager
def __allow_nonbracketed_mutation():
    global __allow_nonbracketed_mutation_flag
    old = __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = True
    try:
        yield
    finally:
        __allow_nonbracketed_mutation_flag = old


class ContextProp:
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter()

    def __set__(self, obj, val):
        if not flags_frozen():
            self.setter(val)
        else:
            raise RuntimeError(
                f"not allowed to set {obj.__name__} flags "
                "after disable_global_flags; please use flags() context manager instead"
            )


class PropModule(types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)


class _FP32Precision:
    def __init__(self, backend, op):
        self.backend = backend
        self.op = op

    def __setattr__(self, name, value):
        if name == "fp32_precision":
            torch._C._set_fp32_precision_setter(self.backend, self.op, value)
        elif name in ("backend", "op"):
            super().__setattr__(name, value)
        else:
            raise AttributeError("Unknown attribute " + name)

    def __getattr__(self, name):
        if name == "fp32_precision":
            return torch._C._get_fp32_precision_getter(self.backend, self.op)
        else:
            raise AttributeError("Unknown attribute " + name)


def set_flags(_fp32_precision="none"):
    orig_flags = (torch._C._get_fp32_precision_getter("generic", "all"),)
    if _fp32_precision is not None:
        torch._C._set_fp32_precision_setter("generic", "all", _fp32_precision)
    return orig_flags


@contextmanager
def flags(fp32_precision="none"):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(fp32_precision)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


def _get_fp32_precision_getter(backend, op):
    def inner():
        return torch._C._get_fp32_precision_getter(backend, op)

    return inner


def _set_fp32_precision_setter(backend, op):
    def inner(precision):
        return torch._C._set_fp32_precision_setter(backend, op, precision)

    return inner


class GenericModule(PropModule):
    fp32_precision = ContextProp(
        _get_fp32_precision_getter("generic", "all"),
        _set_fp32_precision_setter("generic", "all"),
    )


sys.modules[__name__] = GenericModule(sys.modules[__name__], __name__)

from torch.backends import (
    cpu as cpu,
    cuda as cuda,
    cudnn as cudnn,
    cusparselt as cusparselt,
    kleidiai as kleidiai,
    mha as mha,
    miopen as miopen,
    mkl as mkl,
    mkldnn as mkldnn,
    mps as mps,
    nnpack as nnpack,
    openmp as openmp,
    opt_einsum as opt_einsum,
    python_native as python_native,
    quantized as quantized,
)
<|endoftext|># mypy: allow-untyped-defs
import hashlib
import json

import coremltools as ct  # type: ignore[import]
from coremltools.converters.mil.input_types import TensorType  # type: ignore[import]
from coremltools.converters.mil.mil import types  # type: ignore[import]
from coremltools.models.neural_network import quantization_utils  # type: ignore[import]

import torch


CT_METADATA_VERSION = "com.github.apple.coremltools.version"
CT_METADATA_SOURCE = "com.github.apple.coremltools.source"


class ScalarType:
    Float = 0
    Double = 1
    Int = 2
    Long = 3
    Undefined = 4


# Supported Tensor types in coremltools:
# https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/frontend/torch/converter.py#L28
torch_to_mil_types = {
    ScalarType.Float: types.fp32,
    ScalarType.Double: types.fp64,
    ScalarType.Int: types.int32,
    ScalarType.Long: types.int64,
}


class CoreMLComputeUnit:
    CPU = "cpuOnly"
    CPUAndGPU = "cpuAndGPU"
    ALL = "all"


class CoreMLQuantizationMode:
    LINEAR = "linear"
    LINEAR_SYMMETRIC = "linear_symmetric"
    NONE = "none"


def TensorSpec(shape, dtype=ScalarType.Float):
    return (shape, dtype)


def CompileSpec(
    inputs,
    outputs,
    backend=CoreMLComputeUnit.CPU,
    allow_low_precision=True,
    quantization_mode=CoreMLQuantizationMode.NONE,
    mlmodel_export_path=None,
    convert_to=None,
):
    return (
        inputs,
        outputs,
        backend,
        allow_low_precision,
        quantization_mode,
        mlmodel_export_path,
        convert_to,
    )


def _check_enumerated_shape(shape):
    for s in shape:
        if not isinstance(s, (list, tuple)):
            return False
    return True


def _convert_to_mil_type(shape, dtype, name: str):
    mil_shape = shape
    if _check_enumerated_shape(shape):
        mil_shape = ct.EnumeratedShapes(shape)
    ml_type = TensorType(shape=mil_shape, dtype=torch_to_mil_types[dtype])
    ml_type.name = name
    return ml_type


def preprocess(script_module: torch._C.ScriptObject, compile_spec: dict[str, tuple]):
    spec = compile_spec["forward"]
    (
        input_specs,
        output_specs,
        backend,
        allow_low_precision,
        quantization_mode,
        mlmodel_export_path,
        convert_to,
    ) = spec
    mil_inputs = []
    inputs = []
    for index, input in enumerate(input_specs):
        shape, dtype = input
        name = "input_" + str(index)
        inputs.append([name, str(dtype), str(shape)])
        ml_type = _convert_to_mil_type(shape, dtype, name)
        mil_inputs.append(ml_type)
    model = torch.jit.RecursiveScriptModule._construct(script_module, lambda x: None)
    mlmodel = ct.convert(model, inputs=mil_inputs, convert_to=convert_to)

    if quantization_mode != CoreMLQuantizationMode.NONE:
        quant_model_spec = quantization_utils.quantize_weights(
            mlmodel, nbits=8, quantization_mode=quantization_mode
        )
        mlmodel = ct.models.MLModel(quant_model_spec)

    spec = mlmodel.get_spec()
    if len(spec.description.output) != len(output_specs):  # type: ignore[attr-defined]
        raise AssertionError(
            f"Number of outputs in spec ({len(spec.description.output)}) "  # type: ignore[attr-defined]
            f"does not match output_specs ({len(output_specs)})"
        )
    outputs = []
    for index, output in enumerate(output_specs):
        shape, dtype = output
        name = spec.description.output[index].name  # type: ignore[attr-defined]
        outputs.append([name, str(dtype), str(shape)])
    mlmodel = ct.models.model.MLModel(spec)
    print(mlmodel)

    if mlmodel_export_path is not None:
        print(f"Saving CoreML .mlmodel file to {mlmodel_export_path}")
        mlmodel.save(mlmodel_export_path)

    config = {
        "spec_ver": str(spec.specificationVersion),  # type: ignore[attr-defined]
        "backend": backend,
        "allow_low_precision": str(allow_low_precision),
    }
    metadata = {
        "coremltool_ver": mlmodel.user_defined_metadata[CT_METADATA_VERSION],
        "torch_ver": mlmodel.user_defined_metadata[CT_METADATA_SOURCE],
    }
    coreml_compile_spec = {
        "inputs": inputs,
        "outputs": outputs,
        "config": config,
        "metadata": metadata,
    }
    mlmodel = spec.SerializeToString()  # type: ignore[attr-defined]

    return {
        "model": mlmodel,
        "hash": str(hashlib.sha256(mlmodel).hexdigest()),
        "extra": json.dumps(coreml_compile_spec),
    }
<|endoftext|>import torch


__all__ = [
    "get_cpu_capability",
]


def get_cpu_capability() -> str:
    r"""Return cpu capability as a string value.

    Possible values:
    - "DEFAULT"
    - "VSX"
    - "Z VECTOR"
    - "NO AVX"
    - "AVX2"
    - "AVX512"
    - "SVE256"
    """
    return torch._C._get_cpu_capability()
<|endoftext|># mypy: allow-untyped-defs
import sys
import types

import torch


# This function should correspond to the enums present in c10/core/QEngine.h
def _get_qengine_id(qengine: str) -> int:
    if qengine == "none" or qengine == "" or qengine is None:
        ret = 0
    elif qengine == "fbgemm":
        ret = 1
    elif qengine == "qnnpack":
        ret = 2
    elif qengine == "onednn":
        ret = 3
    elif qengine == "x86":
        ret = 4
    else:
        ret = -1
        raise RuntimeError(f"{qengine} is not a valid value for quantized engine")
    return ret


# This function should correspond to the enums present in c10/core/QEngine.h
def _get_qengine_str(qengine: int) -> str:
    all_engines = {0: "none", 1: "fbgemm", 2: "qnnpack", 3: "onednn", 4: "x86"}
    return all_engines.get(qengine, "*undefined")


class _QEngineProp:
    def __get__(self, obj, objtype) -> str:
        return _get_qengine_str(torch._C._get_qengine())

    def __set__(self, obj, val: str) -> None:
        torch._C._set_qengine(_get_qengine_id(val))


class _SupportedQEnginesProp:
    def __get__(self, obj, objtype) -> list[str]:
        qengines = torch._C._supported_qengines()
        return [_get_qengine_str(qe) for qe in qengines]

    def __set__(self, obj, val) -> None:
        raise RuntimeError("Assignment not supported")


class QuantizedEngine(types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)

    engine = _QEngineProp()
    supported_engines = _SupportedQEnginesProp()


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = QuantizedEngine(sys.modules[__name__], __name__)
engine: str
supported_engines: list[str]
<|endoftext|># mypy: allow-untyped-defs
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch.backends import (
    __allow_nonbracketed_mutation,
    _FP32Precision,
    _get_fp32_precision_getter,
    _set_fp32_precision_setter,
    ContextProp,
    PropModule,
)


def is_available():
    r"""Return whether PyTorch is built with MKL-DNN support."""
    return torch._C._has_mkldnn


def is_acl_available():
    r"""Return whether PyTorch is built with MKL-DNN + ACL support."""
    # pyrefly: ignore [missing-attribute]
    return torch._C._has_mkldnn_acl


VERBOSE_OFF = 0
VERBOSE_ON = 1
VERBOSE_ON_CREATION = 2


class verbose:
    """
    On-demand oneDNN (former MKL-DNN) verbosing functionality.

    To make it easier to debug performance issues, oneDNN can dump verbose
    messages containing information like kernel size, input data size and
    execution duration while executing the kernel. The verbosing functionality
    can be invoked via an environment variable named `DNNL_VERBOSE`. However,
    this methodology dumps messages in all steps. Those are a large amount of
    verbose messages. Moreover, for investigating the performance issues,
    generally taking verbose messages for one single iteration is enough.
    This on-demand verbosing functionality makes it possible to control scope
    for verbose message dumping. In the following example, verbose messages
    will be dumped out for the second inference only.

    .. highlight:: python
    .. code-block:: python

        import torch

        model(data)
        with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
            model(data)

    Args:
        level: Verbose level
            - ``VERBOSE_OFF``: Disable verbosing
            - ``VERBOSE_ON``:  Enable verbosing
            - ``VERBOSE_ON_CREATION``: Enable verbosing, including oneDNN kernel creation
    """

    def __init__(self, level):
        self.level = level

    def __enter__(self):
        if self.level == VERBOSE_OFF:
            return
        st = torch._C._verbose.mkldnn_set_verbose(self.level)
        if not st:
            raise AssertionError(
                "Failed to set MKLDNN into verbose mode. Please consider to disable this verbose scope."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch._C._verbose.mkldnn_set_verbose(VERBOSE_OFF)
        return False


def set_flags(
    _enabled=None, _deterministic=None, _allow_tf32=None, _fp32_precision="none"
):
    orig_flags = (
        torch._C._get_mkldnn_enabled(),
        torch._C._get_mkldnn_deterministic(),
        torch._C._get_onednn_allow_tf32(),
        torch._C._get_fp32_precision_getter("mkldnn", "all"),
    )
    if _enabled is not None:
        torch._C._set_mkldnn_enabled(_enabled)
    if _deterministic is not None:
        torch._C._set_mkldnn_deterministic(_deterministic)
    if _allow_tf32 is not None:
        torch._C._set_onednn_allow_tf32(_allow_tf32)
    if _fp32_precision is not None:
        torch._C._set_fp32_precision_setter("mkldnn", "all", _fp32_precision)
    return orig_flags


@contextmanager
def flags(enabled=False, deterministic=False, allow_tf32=True, fp32_precision="none"):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, deterministic, allow_tf32, fp32_precision)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


class MkldnnModule(PropModule):
    def is_available(self):
        return is_available()

    enabled = ContextProp(torch._C._get_mkldnn_enabled, torch._C._set_mkldnn_enabled)
    deterministic = ContextProp(
        torch._C._get_mkldnn_deterministic, torch._C._set_mkldnn_deterministic
    )
    allow_tf32 = ContextProp(
        torch._C._get_onednn_allow_tf32, torch._C._set_onednn_allow_tf32
    )
    matmul = _FP32Precision("mkldnn", "matmul")
    conv = _FP32Precision("mkldnn", "conv")
    rnn = _FP32Precision("mkldnn", "rnn")
    fp32_precision = ContextProp(
        _get_fp32_precision_getter("mkldnn", "all"),
        _set_fp32_precision_setter("generic", "all"),
    )


if TYPE_CHECKING:
    enabled: ContextProp
    deterministic: ContextProp
    allow_tf32: ContextProp
    fp32_precision: str

sys.modules[__name__] = MkldnnModule(sys.modules[__name__], __name__)
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs

from typing import Optional

import torch
from torch.backends._nnapi.serializer import _NnapiSerializer


ANEURALNETWORKS_PREFER_LOW_POWER = 0
ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1
ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2


class NnapiModule(torch.nn.Module):
    """Torch Module that wraps an NNAPI Compilation.

    This module handles preparing the weights, initializing the
    NNAPI TorchBind object, and adjusting the memory formats
    of all inputs and outputs.
    """

    # _nnapi.Compilation is defined
    comp: Optional[torch.classes._nnapi.Compilation]  # type: ignore[name-defined]  # noqa: UP045
    weights: list[torch.Tensor]
    out_templates: list[torch.Tensor]

    def __init__(
        self,
        shape_compute_module: torch.nn.Module,
        ser_model: torch.Tensor,
        weights: list[torch.Tensor],
        inp_mem_fmts: list[int],
        out_mem_fmts: list[int],
        compilation_preference: int,
        relax_f32_to_f16: bool,
    ):
        super().__init__()
        self.shape_compute_module = shape_compute_module
        self.ser_model = ser_model
        self.weights = weights
        self.inp_mem_fmts = inp_mem_fmts
        self.out_mem_fmts = out_mem_fmts
        self.out_templates = []
        self.comp = None
        self.compilation_preference = compilation_preference
        self.relax_f32_to_f16 = relax_f32_to_f16

    @torch.jit.export
    def init(self, args: list[torch.Tensor]):
        if self.comp is not None:
            raise AssertionError("comp must be None before initialization")
        self.out_templates = self.shape_compute_module.prepare(self.ser_model, args)  # type: ignore[operator]
        self.weights = [w.contiguous() for w in self.weights]
        comp = torch.classes._nnapi.Compilation()
        comp.init2(
            self.ser_model,
            self.weights,
            self.compilation_preference,
            self.relax_f32_to_f16,
        )

        self.comp = comp

    def forward(self, args: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.comp is None:
            self.init(args)
        comp = self.comp
        if comp is None:
            raise AssertionError("comp must not be None")
        outs = [torch.empty_like(out) for out in self.out_templates]

        if len(args) != len(self.inp_mem_fmts):
            raise AssertionError(
                f"args length {len(args)} != inp_mem_fmts length {len(self.inp_mem_fmts)}"
            )
        fixed_args = []
        for idx in range(len(args)):
            fmt = self.inp_mem_fmts[idx]
            # These constants match the values in DimOrder in serializer.py
            # TODO: See if it's possible to use those directly.
            if fmt == 0:
                fixed_args.append(args[idx].contiguous())
            elif fmt == 1:
                fixed_args.append(args[idx].permute(0, 2, 3, 1).contiguous())
            else:
                raise ValueError("Invalid mem_fmt")
        comp.run(fixed_args, outs)
        if len(outs) != len(self.out_mem_fmts):
            raise AssertionError(
                f"outs length {len(outs)} != out_mem_fmts length {len(self.out_mem_fmts)}"
            )
        for idx in range(len(self.out_templates)):
            fmt = self.out_mem_fmts[idx]
            # These constants match the values in DimOrder in serializer.py
            # TODO: See if it's possible to use those directly.
            if fmt in (0, 2):
                pass
            elif fmt == 1:
                outs[idx] = outs[idx].permute(0, 3, 1, 2)
            else:
                raise ValueError("Invalid mem_fmt")
        return outs


def convert_model_to_nnapi(
    model,
    inputs,
    serializer=None,
    return_shapes=None,
    use_int16_for_qint16=False,
    compilation_preference=ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
    relax_f32_to_f16=False,
):
    (
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        retval_count,
    ) = process_for_nnapi(
        model, inputs, serializer, return_shapes, use_int16_for_qint16
    )

    nnapi_model = NnapiModule(
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        compilation_preference,
        relax_f32_to_f16,
    )

    class NnapiInterfaceWrapper(torch.nn.Module):
        """NNAPI list-ifying and de-list-ifying wrapper.

        NNAPI always expects a list of inputs and provides a list of outputs.
        This module allows us to accept inputs as separate arguments.
        It returns results as either a single tensor or tuple,
        matching the original module.
        """

        def __init__(self, mod):
            super().__init__()
            self.mod = mod

    wrapper_model_py = NnapiInterfaceWrapper(nnapi_model)
    wrapper_model = torch.jit.script(wrapper_model_py)
    # TODO: Maybe make these names match the original.
    arg_list = ", ".join(f"arg_{idx}" for idx in range(len(inputs)))
    if retval_count < 0:
        ret_expr = "retvals[0]"
    else:
        ret_expr = "".join(f"retvals[{idx}], " for idx in range(retval_count))
    wrapper_model.define(
        f"def forward(self, {arg_list}):\n"
        f"    retvals = self.mod([{arg_list}])\n"
        f"    return {ret_expr}\n"
    )
    return wrapper_model


def process_for_nnapi(
    model, inputs, serializer=None, return_shapes=None, use_int16_for_qint16=False
):
    model = torch.jit.freeze(model)

    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    serializer = serializer or _NnapiSerializer(
        config=None, use_int16_for_qint16=use_int16_for_qint16
    )
    (
        ser_model,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        shape_compute_lines,
        retval_count,
    ) = serializer.serialize_model(model, inputs, return_shapes)
    ser_model_tensor = torch.tensor(ser_model, dtype=torch.int32)

    # We have to create a new class here every time this function is called
    # because module.define adds a method to the *class*, not the instance.
    class ShapeComputeModule(torch.nn.Module):
        """Code-gen-ed module for tensor shape computation.

        module.prepare will mutate ser_model according to the computed operand
        shapes, based on the shapes of args.  Returns a list of output templates.
        """

    shape_compute_module = torch.jit.script(ShapeComputeModule())
    real_shape_compute_lines = [
        "def prepare(self, ser_model: torch.Tensor, args: List[torch.Tensor]) -> List[torch.Tensor]:\n",
    ] + [f"    {line}\n" for line in shape_compute_lines]
    shape_compute_module.define("".join(real_shape_compute_lines))

    return (
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        retval_count,
    )
<|endoftext|># mypy: allow-untyped-defs
import array
import enum
import functools
import logging
import operator
import struct
import sys
from typing import NamedTuple

import torch


# TODO: Add type annotations
# TODO: Check tensor types for ops


LOG = logging.getLogger("nnapi_serialize")


class NNAPI_OperandCode:
    FLOAT32 = 0
    INT32 = 1
    UINT32 = 2
    TENSOR_FLOAT32 = 3
    TENSOR_INT32 = 4
    TENSOR_QUANT8_ASYMM = 5
    BOOL = 6
    TENSOR_QUANT16_SYMM = 7
    TENSOR_FLOAT16 = 8
    TENSOR_BOOL8 = 9
    FLOAT16 = 10
    TENSOR_QUANT8_SYMM_PER_CHANNEL = 11
    TENSOR_QUANT16_ASYMM = 12


class NNAPI_OperationCode:
    ADD = 0
    AVERAGE_POOL_2D = 1
    CONCATENATION = 2
    CONV_2D = 3
    DEPTHWISE_CONV_2D = 4
    DEPTH_TO_SPACE = 5
    DEQUANTIZE = 6
    EMBEDDING_LOOKUP = 7
    FLOOR = 8
    FULLY_CONNECTED = 9
    HASHTABLE_LOOKUP = 10
    L2_NORMALIZATION = 11
    L2_POOL_2D = 12
    LOCAL_RESPONSE_NORMALIZATION = 13
    LOGISTIC = 14
    LSH_PROJECTION = 15
    LSTM = 16
    MAX_POOL_2D = 17
    MUL = 18
    RELU = 19
    RELU1 = 20
    RELU6 = 21
    RESHAPE = 22
    RESIZE_BILINEAR = 23
    RNN = 24
    SOFTMAX = 25
    SPACE_TO_DEPTH = 26
    SVDF = 27
    TANH = 28
    BATCH_TO_SPACE_ND = 29
    DIV = 30
    MEAN = 31
    PAD = 32
    SPACE_TO_BATCH_ND = 33
    SQUEEZE = 34
    STRIDED_SLICE = 35
    SUB = 36
    TRANSPOSE = 37
    ABS = 38
    ARGMAX = 39
    ARGMIN = 40
    AXIS_ALIGNED_BBOX_TRANSFORM = 41
    BIDIRECTIONAL_SEQUENCE_LSTM = 42
    BIDIRECTIONAL_SEQUENCE_RNN = 43
    BOX_WITH_NMS_LIMIT = 44
    CAST = 45
    CHANNEL_SHUFFLE = 46
    DETECTION_POSTPROCESSING = 47
    EQUAL = 48
    EXP = 49
    EXPAND_DIMS = 50
    GATHER = 51
    GENERATE_PROPOSALS = 52
    GREATER = 53
    GREATER_EQUAL = 54
    GROUPED_CONV_2D = 55
    HEATMAP_MAX_KEYPOINT = 56
    INSTANCE_NORMALIZATION = 57
    LESS = 58
    LESS_EQUAL = 59
    LOG = 60
    LOGICAL_AND = 61
    LOGICAL_NOT = 62
    LOGICAL_OR = 63
    LOG_SOFTMAX = 64
    MAXIMUM = 65
    MINIMUM = 66
    NEG = 67
    NOT_EQUAL = 68
    PAD_V2 = 69
    POW = 70
    PRELU = 71
    QUANTIZE = 72
    QUANTIZED_16BIT_LSTM = 73
    RANDOM_MULTINOMIAL = 74
    REDUCE_ALL = 75
    REDUCE_ANY = 76
    REDUCE_MAX = 77
    REDUCE_MIN = 78
    REDUCE_PROD = 79
    REDUCE_SUM = 80
    ROI_ALIGN = 81
    ROI_POOLING = 82
    RSQRT = 83
    SELECT = 84
    SIN = 85
    SLICE = 86
    SPLIT = 87
    SQRT = 88
    TILE = 89
    TOPK_V2 = 90
    TRANSPOSE_CONV_2D = 91
    UNIDIRECTIONAL_SEQUENCE_LSTM = 92
    UNIDIRECTIONAL_SEQUENCE_RNN = 93
    RESIZE_NEAREST_NEIGHBOR = 94


class NNAPI_FuseCode:
    FUSED_NONE = 0
    FUSED_RELU = 1
    FUSED_RELU1 = 2
    FUSED_RELU6 = 3


class OperandValueSourceType:
    IMMEDIATE = 0
    NUMBERED_BUFFER = 2
    NUMBERED_MEMORY = 3


# Scalar types that appear explicitly in models.
# These must be kept in sync with
# AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS.
# TODO: Expose these directly to Python to avoid maintaining this list.
class TorchScalarTypes(enum.Enum):
    QUINT8 = 13


def approx_equal(lhs, rhs, tolerance=1e-6):
    return abs(lhs - rhs) <= tolerance * min(lhs, rhs)


def tensor_size(op_type, dims):
    ITEM_SIZES = {
        NNAPI_OperandCode.TENSOR_FLOAT32: 4,
        NNAPI_OperandCode.TENSOR_INT32: 4,
        NNAPI_OperandCode.TENSOR_QUANT8_ASYMM: 1,
        NNAPI_OperandCode.TENSOR_QUANT16_SYMM: 2,
        NNAPI_OperandCode.TENSOR_QUANT16_ASYMM: 2,
    }
    size = ITEM_SIZES[op_type]
    for d in dims:
        size *= d
    return size


def change_element(tup, index, value):
    ls = list(tup)
    ls[index] = value
    return tuple(ls)


class ConvPoolArgs2d(NamedTuple):
    """Configuration arguments for a convolution."""

    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_t: int
    pad_b: int
    pad_l: int
    pad_r: int
    dilation_h: int
    dilation_w: int
    group: int


class DimOrder(enum.Enum):
    PRESUMED_CONTIGUOUS = 0
    CHANNELS_LAST = 1
    SCALAR_OR_VECTOR = 2
    UNKNOWN_CONSTANT = 999


class Operand(NamedTuple):
    """Representation of an NNAPI operand."""

    # NNAPI operand type.  One of NNAPI_OperandCode.
    # TODO: Make this an enum.
    op_type: int

    # This is always the PyTorch shape, which is NCHW for feature maps.
    # The actual NNAPI operand might have a transposed shape.
    # we use 0 for load time dynamic shapes & -1 for runtime dynamic shapes
    shape: tuple[int, ...]

    # Specifies how the shape of the operand that we define in NNAPI
    # relates to the shape we track above.
    # - PRESUMED_CONTIGUOUS: physical NNAPI operand will exactly match
    #   the shape of the PyTorch tensor.
    # - CHANNELS_LAST: The PyTorch tensor is expected to be NCHW, and
    #   the NNAPI operand will be represented explicitly as NHWC.
    dim_order: DimOrder

    # Quantization params
    scale: float
    zero_point: int

    def use_nchw(self):
        if self.dim_order is DimOrder.PRESUMED_CONTIGUOUS:
            return True
        if self.dim_order is DimOrder.CHANNELS_LAST:
            return False
        raise Exception("Unknown dim order")  # noqa: TRY002


def broadcast_shapes(shape1, shape2):
    if len(shape1) <= 0:
        raise AssertionError(f"shape1 must have length > 0, got {len(shape1)}")
    if len(shape2) <= 0:
        raise AssertionError(f"shape2 must have length > 0, got {len(shape2)}")
    s1 = list(shape1)
    s2 = list(shape2)
    # TODO: Support non-equal-rank broadcast where semantics match.
    # This can be tricky for NHWC tensors because dimension orders
    # don't match between PT and NNAPI, even though semantics match.
    if len(s1) > len(s2):
        # s2 = [1] * (len(s1) - len(s2)) + s2
        raise Exception(  # noqa: TRY002
            "Non-equal-rank broadcast is not supported yet."
        )
    if len(s2) > len(s1):
        # s3 = [1] * (len(s2) - len(s1)) + s1
        raise Exception(  # noqa: TRY002
            "Non-equal-rank broadcast is not supported yet."
        )
    ret = []
    for d1, d2 in zip(s1, s2):
        if d1 == 1:
            ret.append(d2)
        elif d2 == 1:
            ret.append(d1)
        elif d1 == d2:
            ret.append(d1)
        else:
            raise Exception(  # noqa: TRY002
                f"Cannot broadcast shapes: {shape1} and {shape2}"
            )
    return tuple(ret)


def get_conv_pool_shape(image_shape, args, out_ch, transpose):
    batch, _in_c, in_h, in_w = image_shape

    # TODO: Handle dilation
    if args.dilation_h != 1 or args.dilation_w != 1:
        raise Exception("Dilation not supported yet.")  # noqa: TRY002

    if transpose:
        out_h = (in_h - 1) * args.stride_h + args.kernel_h - args.pad_t - args.pad_b
        out_w = (in_w - 1) * args.stride_w + args.kernel_w - args.pad_l - args.pad_l
    else:
        out_h = (in_h - args.kernel_h + args.pad_t + args.pad_b) // args.stride_h + 1
        out_w = (in_w - args.kernel_w + args.pad_l + args.pad_r) // args.stride_w + 1

    # Handle variable-sized tensors.
    if in_h == 0:
        out_h = 0
    if in_w == 0:
        out_w = 0

    out_shape = (batch, out_ch, out_h, out_w)
    return out_shape


def fix_shape(shape, dim_order):
    # Return the actual shape that an operand should have in NNAPI,
    # given a PyTorch shape and dimension order.  This is where we
    # convert from PyTorch's "always NCHW" shape to explicit NHWC.
    if dim_order is DimOrder.PRESUMED_CONTIGUOUS:
        return shape
    if dim_order is DimOrder.CHANNELS_LAST:
        return tuple([shape[0]] + list(shape[2:]) + [shape[1]])
    if dim_order is DimOrder.SCALAR_OR_VECTOR:
        if not (len(shape) == 0 or len(shape) == 1):
            raise AssertionError(
                f"SCALAR_OR_VECTOR requires len(shape) == 0 or 1, got {len(shape)}"
            )
        return shape
    if dim_order is DimOrder.UNKNOWN_CONSTANT:
        # XXX think this through
        return shape
    raise Exception(f"Bad dim_order: {dim_order!r}.")  # noqa: TRY002


def reverse_map_dim(dim_order, d):
    # Return the original PyTorch dimension position for a given dimension.
    # d should be the dimension that NNAPI will see.
    # reverse_map_dim(PRESUMED_CONTIGUOUS, x) == x
    # reverse_map_dim(CHANNELS_LAST, 3) == 1
    if dim_order in (DimOrder.PRESUMED_CONTIGUOUS, DimOrder.SCALAR_OR_VECTOR):
        return d
    if dim_order is DimOrder.CHANNELS_LAST:
        return [0, 2, 3, 1][d]
    raise AssertionError(f"expected DimOrder.CHANNELS_LAST, got {dim_order}")


def flex_name(op_id, dim):
    # Return the local variable name for the computed flexible size
    # for a given op and dimension.
    return f"s_{op_id}_{dim}"


class _NnapiSerializer:
    def __init__(self, config, use_int16_for_qint16=False):
        self.operands = []
        self.values = []
        self.operations = []
        self.value_data = []
        self.operation_args = []
        self.inputs = []
        self.outputs = []
        self.flexible_shape_computation_lines = []

        self.modules = {}
        self.constants = {}
        self.tensor_sequences = {}
        self.jitval_operand_map = {}
        self.cached_immediates = {}
        self.used_weights = []
        self.weight_offset = 0
        self.use_int16_for_qint16 = use_int16_for_qint16

        if config is None:
            config = {}

    def get_next_operand_id(self):
        return len(self.operands)

    # Add a tensor operand corresponding to a JIT Value.
    # Returns the NNAPI operand ID.  Can be looked up later with
    # get_tensor_operand_by_jitval.
    def add_tensor_operand(self, jitval, oper):
        if not isinstance(oper, Operand):
            raise AssertionError(f"expected Operand, got {type(oper)}")
        if jitval in self.jitval_operand_map:
            raise Exception(f"Duplicate tensor: {jitval!r}")  # noqa: TRY002

        operand_id = self.get_next_operand_id()
        self.operands.append(oper)
        self.jitval_operand_map[jitval] = operand_id
        return operand_id

    # Add a tensor operand that does not correspond to a JIT Value.
    # Useful for cases where multiple NNAPI operands are required
    # to implement one JIT IR node.  Returns the NNAPI operand ID.
    def add_anonymous_tensor_operand(self, oper):
        if not isinstance(oper, Operand):
            raise AssertionError(f"expected Operand, got {type(oper)}")
        operand_id = self.get_next_operand_id()
        self.operands.append(oper)
        return operand_id

    def torch_tensor_to_operand(self, tensor, dim_order):
        dtype = str(tensor.dtype).replace("torch.", "")
        scale = 0.0
        zero_point = 0
        if dtype == "float32":
            op_type = NNAPI_OperandCode.TENSOR_FLOAT32
        elif dtype == "int32":
            op_type = NNAPI_OperandCode.TENSOR_INT32
        elif dtype == "quint8":
            op_type = NNAPI_OperandCode.TENSOR_QUANT8_ASYMM
            scale = tensor.q_scale()
            zero_point = tensor.q_zero_point()
        elif dtype == "qint32":
            op_type = NNAPI_OperandCode.TENSOR_INT32
            scale = tensor.q_scale()
            zero_point = tensor.q_zero_point()
            if zero_point != 0:
                raise AssertionError(f"qint32 zero_point must be 0, got {zero_point}")
        elif dtype == "int16":
            if self.use_int16_for_qint16:
                nnapi_dtype = getattr(tensor, "nnapi_dtype", None)
                op_codes = (
                    NNAPI_OperandCode.TENSOR_QUANT16_SYMM,
                    NNAPI_OperandCode.TENSOR_QUANT16_ASYMM,
                )
                if nnapi_dtype in op_codes:
                    op_type = nnapi_dtype
                    scale = tensor.nnapi_scale
                    zero_point = tensor.nnapi_zero_point
                else:
                    raise Exception(  # noqa: TRY002
                        f"`nnapi_type` needs to be one of {op_codes} for `int16`"
                    )
            else:
                raise Exception(  # noqa: TRY002
                    "`int16` isn't supported. If you're trying to represent NNAPI"
                    " qint16 with Pytorch int16, set `use_int16_for_qint16 = True`"
                )
        else:
            raise Exception(  # noqa: TRY002
                f"Can't handle input with dtype '{tensor.dtype}'"
            )
        return Operand(
            shape=tuple(tensor.shape),
            # pyrefly: ignore [bad-argument-type]
            op_type=op_type,
            dim_order=dim_order,
            scale=scale,
            zero_point=zero_point,
        )

    def add_tensor_operand_for_input(self, arg_idx, jitval, tensor):
        dim_order = (
            DimOrder.CHANNELS_LAST
            if getattr(tensor, "nnapi_nhwc", False)
            else DimOrder.PRESUMED_CONTIGUOUS
        )
        toper = self.torch_tensor_to_operand(tensor, dim_order)
        operand_id = self.add_tensor_operand(jitval, toper)
        self.inputs.append(operand_id)
        for dim, size in enumerate(tensor.shape):
            if size == 0:
                self.compute_operand_shape(
                    operand_id, dim, f"args[{arg_idx}].shape[{dim}]"
                )
        return operand_id

    def add_tensor_operand_for_weight(
        self, tensor, dim_order=DimOrder.UNKNOWN_CONSTANT
    ):
        toper = self.torch_tensor_to_operand(tensor, dim_order)
        operand_id = len(self.operands)
        self.operands.append(toper)
        tsize = tensor_size(toper.op_type, toper.shape)
        self.values.append((operand_id, OperandValueSourceType.NUMBERED_BUFFER))
        buf_num = len(self.used_weights)
        offset = 0
        self.value_data.append(struct.pack("iii", buf_num, offset, tsize))
        # For NHWC NNAPI op, lay out data in the same dim order by permuting torch tensor
        if dim_order == DimOrder.CHANNELS_LAST:
            tensor = tensor.permute(0, 2, 3, 1)
        self.used_weights.append(tensor)
        return operand_id

    def add_immediate_operand(self, code, value, dims):
        if not isinstance(dims, tuple):
            raise AssertionError(f"dims must be a tuple, got {type(dims)}")
        cache_key = (code, value)
        if cache_key not in self.cached_immediates:
            operand_id = len(self.operands)
            self.operands.append(Operand(code, dims, DimOrder.SCALAR_OR_VECTOR, 0.0, 0))
            self.values.append((operand_id, OperandValueSourceType.IMMEDIATE))
            self.value_data.append(value)
            self.cached_immediates[cache_key] = operand_id
        return self.cached_immediates[cache_key]

    def add_immediate_int_scalar(self, value):
        return self.add_immediate_operand(
            NNAPI_OperandCode.INT32, struct.pack("i", value), ()
        )

    def add_immediate_float_scalar(self, value):
        return self.add_immediate_operand(
            NNAPI_OperandCode.FLOAT32, struct.pack("f", value), ()
        )

    def add_immediate_bool_scalar(self, value):
        return self.add_immediate_operand(
            NNAPI_OperandCode.BOOL, b"\x01" if value else b"\x00", ()
        )

    def add_immediate_int_vector(self, value):
        return self.add_immediate_operand(
            NNAPI_OperandCode.TENSOR_INT32,
            array.array("i", value).tobytes(),
            (len(value),),
        )

    def has_operand_for_jitval(self, jitval):
        return jitval in self.jitval_operand_map

    def get_tensor_operand_by_jitval(self, jitval):
        operand_id = self.jitval_operand_map[jitval]
        return (operand_id, self.operands[operand_id])

    def get_tensor_operand_by_jitval_fixed_size(self, jitval):
        op_id, oper = self.get_tensor_operand_by_jitval(jitval)
        for s in oper.shape:
            if s == 0:
                # TODO: Improve this error message, possibly after converting
                # many callsites to support flexible size.
                raise Exception(  # noqa: TRY002
                    "Flexible size is not supported for this operand."
                )
            if s < 0:
                # runtime flex
                LOG.warning("Operand %s has runtime flex shape", oper)
        return op_id, oper

    def get_tensor_operand_or_constant(
        self, jitval, dim_order=DimOrder.PRESUMED_CONTIGUOUS
    ):
        operand_id = self.jitval_operand_map.get(jitval)
        if operand_id is None:
            _, value = self.get_constant_value(jitval, "TensorType")
            operand_id = self.add_tensor_operand_for_weight(value, dim_order)
        return (operand_id, self.operands[operand_id])

    def get_tensor_operand_for_weight(self, jitval):
        _, value = self.get_constant_value(jitval, "TensorType")
        operand_id = self.add_tensor_operand_for_weight(value)
        return (operand_id, self.operands[operand_id])

    def add_operation(self, opcode, inputs, outputs):
        self.operations.append((opcode, len(inputs), len(outputs)))
        self.operation_args.extend(inputs + outputs)

    def add_tensor_sequence(self, jitval, values):
        if jitval in self.tensor_sequences:
            raise AssertionError(f"jitval {jitval!r} already in tensor_sequences")
        self.tensor_sequences[jitval] = values

    def add_constant_value(self, jitval, ctype, value):
        if jitval in self.constants:
            raise AssertionError(f"jitval {jitval!r} already in constants")
        self.constants[jitval] = (ctype, value)

    def get_constant_value(self, jitval, typekind=None):
        record = self.constants.get(jitval)
        if record is None:
            raise Exception(  # noqa: TRY002
                f"Could not find constant value for '{jitval!r}'."
            )
        ctype, _ = record
        if typekind is not None and ctype.kind() != typekind:
            raise Exception(  # noqa: TRY002
                f"Expected constant value of type {typekind}, but got {ctype.kind()} for value '{jitval!r}'"
            )
        return record

    def operand_to_template_torchscript(self, op_id, oper, shape=None):
        """Return a TorchScript expression to build a template for a given operand."""
        if shape is None:
            shape = oper.shape
        else:
            if len(shape) != len(oper.shape):
                raise AssertionError(
                    f"shape length {len(shape)} != oper.shape length {len(oper.shape)}"
                )

        shape_parts = ["("]
        for d, s in enumerate(shape):
            if s > 0:
                # Fixed shape dimension: just add the value.
                shape_parts.append(str(s))
            elif s == 0:
                # Load time flexible shape dimension: it should have been computed in a variable.
                shape_parts.append(flex_name(op_id, d))
            elif s == -1:
                # Runtime flexible shape
                shape_parts.append("0")
            else:
                raise Exception(  # noqa: TRY002
                    "Unknown dim value, dimensions should be >= -1"
                )
            shape_parts.append(",")
        shape_parts.append(")")
        shape_code = "".join(shape_parts)
        if oper.op_type == NNAPI_OperandCode.TENSOR_FLOAT32:
            return f"torch.zeros({shape_code}, dtype=torch.float32)"
        elif oper.op_type == NNAPI_OperandCode.TENSOR_INT32:
            return f"torch.zeros({shape_code}, dtype=torch.int32)"
        elif oper.op_type == NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
            return (
                f"torch.quantize_per_tensor("
                f"torch.zeros(1), scale={oper.scale}, zero_point={oper.zero_point}, dtype=torch.quint8)"
                f".expand({shape_code}).contiguous()"
            )
        elif oper.op_type in (
            NNAPI_OperandCode.TENSOR_QUANT16_ASYMM,
            NNAPI_OperandCode.TENSOR_QUANT16_SYMM,
        ):
            if self.use_int16_for_qint16:
                return f"torch.zeros({shape_code}, dtype=torch.int16)"
            else:
                raise Exception(  # noqa: TRY002
                    "`int16` isn't supported. If you're trying to represent NNAPI"
                    " qint16 with Pytorch int16, set `use_int16_for_qint16 = True`"
                )

        raise Exception(  # noqa: TRY002
            f"Unsupported output operand type: {oper.op_type}"
        )

    def forward_operand_shape(self, out_op_id, out_dim, in_op_id, in_dim):
        self.compute_operand_shape(out_op_id, out_dim, flex_name(in_op_id, in_dim))

    def compute_operand_shape(self, op_id, dim, expr):
        self.flexible_shape_computation_lines.append(
            f"{flex_name(op_id, dim)} = {expr}"
        )

    def transpose_to_nhwc(self, in_id, oper):
        if oper.shape[2:] != (1, 1):
            raise Exception(  # noqa: TRY002
                "Automatic transpose only supported for H,W == 1,1"
            )

        out_oper = oper._replace(dim_order=DimOrder.CHANNELS_LAST)

        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector([0, 2, 3, 1])

        outputs = [None] * 1
        outputs[0] = self.add_anonymous_tensor_operand(out_oper)

        self.add_operation(NNAPI_OperationCode.TRANSPOSE, inputs, outputs)

        return outputs[0], out_oper

    # Transpose inputs as necessary to allow broadcasting.
    def transpose_for_broadcast(self, in0_id, in0_oper, in1_id, in1_oper):
        if in0_oper.dim_order == in1_oper.dim_order:
            return in0_id, in0_oper, in1_id, in1_oper

        # Assume NHWC is preferred if there is a mismatch.
        orders = (in0_oper.dim_order, in1_oper.dim_order)
        if orders == (DimOrder.PRESUMED_CONTIGUOUS, DimOrder.CHANNELS_LAST):
            return self.transpose_to_nhwc(in0_id, in0_oper) + (in1_id, in1_oper)
        if orders == (DimOrder.CHANNELS_LAST, DimOrder.PRESUMED_CONTIGUOUS):
            return (in0_id, in0_oper) + self.transpose_to_nhwc(in1_id, in1_oper)

        raise Exception(  # noqa: TRY002
            f"Automatic transpose not supported for dim_orders: {in0_oper.dim_order!r}, {in1_oper.dim_order!r}"
        )

    def get_size_arg(self, jitval):
        ctype, value = self.get_constant_value(jitval)
        if ctype.kind() == "ListType":
            if ctype.getElementType().kind() != "IntType":
                raise AssertionError(
                    f"expected ListType of IntType, got {ctype.getElementType().kind()}"
                )
            return value
        raise Exception(  # noqa: TRY002
            f"Can't handle size arg of type '{ctype!r}' for '{jitval!r}'"
        )

    def get_conv_pool_args_2d_from_pack(self, kernel_size, packed_config):
        pc = [i.item() for i in packed_config]
        if pc[0] != 2:
            raise AssertionError(f"expected pc[0] == 2, got {pc[0]}")
        strides = [pc[1], pc[2]]
        paddings = [pc[3], pc[4]]
        dilations = [pc[5], pc[6]]
        output_padding = [pc[7], pc[8]]
        group_num = pc[9]

        if len(pc) != 11:
            raise AssertionError(f"expected len(pc) == 11, got {len(pc)}")
        if output_padding != [0, 0]:
            raise AssertionError(
                f"expected output_padding == [0, 0], got {output_padding}"
            )

        return self.get_conv_pool_args_2d_common(
            kernel_size, strides, paddings, dilations, group_num
        )

    def get_conv_pool_args_2d_from_jit(
        self, kernel_size, stride, padding, dilation=None, group=None
    ):
        strides = self.get_size_arg(stride)
        paddings = self.get_size_arg(padding)
        if dilation is None:
            dilations = [1, 1]
        else:
            dilations = self.get_size_arg(dilation)
        if group is not None:
            _, group_num = self.get_constant_value(group, "IntType")
        else:
            group_num = None
        return self.get_conv_pool_args_2d_common(
            kernel_size, strides, paddings, dilations, group_num
        )

    def get_conv_pool_args_2d_common(
        self, kernel_size, strides, paddings, dilations, group_num
    ):
        kernels = list(kernel_size)

        if len(kernels) != 2:
            raise AssertionError(f"expected len(kernels) == 2, got {len(kernels)}")
        if len(strides) != 2:
            raise AssertionError(f"expected len(strides) == 2, got {len(strides)}")
        if len(paddings) != 2:
            raise AssertionError(f"expected len(paddings) == 2, got {len(paddings)}")
        if len(dilations) != 2:
            raise AssertionError(f"expected len(dilations) == 2, got {len(dilations)}")

        # NNAPI uses 4 values for padding.
        ph, pw = paddings
        real_paddings = [ph, ph, pw, pw]

        return ConvPoolArgs2d(
            *(kernels + strides + real_paddings + dilations + [group_num])
        )

    def serialize_model(self, model, inputs, return_shapes=None):
        self.add_immediate_bool_scalar(False)
        self.add_immediate_bool_scalar(True)

        inp_dim_orders = []
        out_dim_orders = []

        self_jitval = next(model.graph.inputs())
        self.add_constant_value(self_jitval, self_jitval.type(), model)

        for arg_idx, (input_value, input_tensor) in enumerate(
            zip(list(model.graph.inputs())[1:], inputs)
        ):
            op_id = self.add_tensor_operand_for_input(
                arg_idx, input_value, input_tensor
            )
            inp_dim_orders.append(self.operands[op_id].dim_order.value)

        for idx, node in enumerate(model.graph.nodes()):
            LOG.debug("Processing node #%d: %r", idx, node)
            self.add_node(node)

        retn = model.graph.return_node()
        if retn.inputsSize() != 1:
            raise AssertionError(
                f"expected retn.inputsSize() == 1, got {retn.inputsSize()}"
            )
        if retn.outputsSize() != 0:
            raise AssertionError(
                f"expected retn.outputsSize() == 0, got {retn.outputsSize()}"
            )
        retn_input = retn.inputsAt(0)
        template_return_lines = ["return ["]
        if retn_input.type().kind() == "TensorType":
            return_values = [retn_input]
            retval_count = -1
        elif retn_input.type().kind() == "TupleType":
            return_values = self.tensor_sequences[retn_input]
            retval_count = len(return_values)
        else:
            raise Exception(  # noqa: TRY002
                f"Unsupported return type: {retn_input.type()}"
            )

        if return_shapes is not None:
            if len(return_shapes) != len(return_values):
                raise AssertionError(
                    f"return_shapes length {len(return_shapes)} != return_values length {len(return_values)}"
                )
        for i, v in enumerate(return_values):
            op_id = self.jitval_operand_map[v]
            self.outputs.append(op_id)
            out_dim_orders.append(self.operands[op_id].dim_order.value)
            shape = return_shapes[i] if return_shapes else None
            template_return_lines.append(
                self.operand_to_template_torchscript(op_id, self.operands[op_id], shape)
                + ","
            )
        template_return_lines.append("]")

        model = []

        version = 1
        header = struct.pack(
            "iiiiii",
            version,
            len(self.operands),
            len(self.values),
            len(self.operations),
            len(self.inputs),
            len(self.outputs),
        )
        model.append(header)

        serialized_values, serialized_value_data = self.serialize_values()

        model.extend(
            struct.pack("iifi", t, len(d), s, z) for (t, d, _m, s, z) in self.operands
        )
        model.extend(serialized_values)
        model.extend(struct.pack("iii", *x) for x in self.operations)

        # Compact the model so we can get its length so far.
        model = [b"".join(model)]
        model_offset = len(model[0])
        # Model offset is the index into the model (in 32-bit words, not bytes)
        # of the next dimension we're about to serialize.  If it's 0,
        # generate code to mutate it before passing to NNAPI.
        if model_offset % 4 != 0:
            raise AssertionError(
                f"model_offset must be divisible by 4, got {model_offset}"
            )
        model_offset = int(model_offset / 4)

        for op_id, (_, dims, dim_order, _, _) in enumerate(self.operands):
            shape = fix_shape(dims, dim_order)
            for d, s in enumerate(shape):
                if s == 0:
                    pt_d = reverse_map_dim(dim_order, d)
                    self.flexible_shape_computation_lines.append(
                        f"ser_model[{model_offset}] = {flex_name(op_id, pt_d)}"
                    )
                model_offset += 1

            # convert runtime flex shape from -1 to 0
            shape = tuple(d if d != -1 else 0 for d in shape)
            model.append(self.serialize_ints(shape))

        model.extend(serialized_value_data)
        model.append(self.serialize_ints(self.operation_args))
        model.append(self.serialize_ints(self.inputs))
        model.append(self.serialize_ints(self.outputs))

        self.flexible_shape_computation_lines.extend(template_return_lines)

        return (
            array.array("i", b"".join(model)),
            self.used_weights,
            inp_dim_orders,
            out_dim_orders,
            self.flexible_shape_computation_lines,
            retval_count,
        )

    def serialize_values(self):
        serialized_values = []
        serialized_value_data = []
        if len(self.values) != len(self.value_data):
            raise AssertionError(
                f"values length {len(self.values)} != value_data length {len(self.value_data)}"
            )
        for (op_index, source_type), data in zip(self.values, self.value_data):
            source_length = len(data)

            # Pad with 0 bytes out to a multiple of 4 for alignment.
            physical_length = ((source_length - 1) | 0x3) + 1
            padded_data = data + (b"\0" * (physical_length - source_length))

            serialized_values.append(
                struct.pack("iii", op_index, source_type, source_length)
            )
            serialized_value_data.append(padded_data)

        return serialized_values, serialized_value_data

    @staticmethod
    def serialize_ints(ints):
        return array.array("i", ints).tobytes()

    ADDER_MAP = {
        "prim::GetAttr": lambda self, node: self.add_getattr(node),
        "prim::Constant": lambda self, node: self.add_constant_node(node),
        "prim::ListConstruct": lambda self, node: self.add_list_construct(node),
        "prim::TupleConstruct": lambda self, node: self.add_tuple_construct(node),
        "aten::unsqueeze": lambda self, node: self.add_unsqueeze(node),
        "aten::to": lambda self, node: self.add_to(node),
        "aten::detach": lambda self, node: self._identity(node),
        "aten::reshape": lambda self, node: self.add_reshape(node),
        "aten::flatten": lambda self, node: self.add_flatten(node),
        "aten::slice": lambda self, node: self.add_slice(node),
        "aten::size": lambda self, node: self.add_size(node),
        "aten::cat": lambda self, node: self.add_cat(node),
        "aten::mean": lambda self, node: self.add_mean(node),
        "aten::quantize_per_tensor": lambda self, node: self.add_quantize(node),
        "aten::dequantize": lambda self, node: self.add_dequantize(node),
        "aten::add": lambda self, node: self.add_add_sub_op(
            node, NNAPI_OperationCode.ADD, NNAPI_FuseCode.FUSED_NONE
        ),
        "aten::sub": lambda self, node: self.add_add_sub_op(
            node, NNAPI_OperationCode.SUB, NNAPI_FuseCode.FUSED_NONE
        ),
        "aten::mul": lambda self, node: self.add_pointwise_simple_binary_broadcast_op(
            node, NNAPI_OperationCode.MUL, NNAPI_FuseCode.FUSED_NONE
        ),
        "aten::div": lambda self, node: self.add_pointwise_simple_binary_broadcast_op(
            node, NNAPI_OperationCode.DIV, NNAPI_FuseCode.FUSED_NONE
        ),
        "aten::relu": lambda self, node: self.add_pointwise_simple_unary_op(
            node, NNAPI_OperationCode.RELU
        ),
        "aten::sigmoid": lambda self, node: self.add_pointwise_simple_unary_op(
            node, NNAPI_OperationCode.LOGISTIC
        ),
        "aten::softmax": lambda self, node: self.add_softmax(node),
        "aten::hardtanh": lambda self, node: self.add_hardtanh(node),
        "aten::avg_pool2d": lambda self, node: self.add_avg_pool2d(node),
        "aten::max_pool2d": lambda self, node: self.add_pool2d_node(
            node, NNAPI_OperationCode.MAX_POOL_2D
        ),
        "aten::adaptive_avg_pool2d": lambda self, node: self.add_adaptive_avg_pool2d(
            node
        ),
        "aten::upsample_nearest2d": lambda self, node: self.add_upsample_nearest2d(
            node
        ),
        "aten::prelu": lambda self, node: self.add_prelu_op(node),
        "aten::addmm": lambda self, node: self.add_addmm(node),
        "aten::linear": lambda self, node: self.add_linear(node),
        "aten::_convolution": lambda self, node: self.add_conv_underscore(node),
        "aten::conv2d": lambda self, node: self.add_conv2d(node),
        "aten::log_softmax": lambda self, node: self.add_log_softmax(node),
        "quantized::linear": lambda self, node: self.add_qlinear(node),
        "quantized::conv2d": lambda self, node: self.add_qconv2d(
            node, NNAPI_FuseCode.FUSED_NONE
        ),
        "quantized::conv2d_relu": lambda self, node: self.add_qconv2d(
            node, NNAPI_FuseCode.FUSED_RELU
        ),
        "quantized::conv_transpose2d": lambda self, node: self.add_qconv2d(
            node, NNAPI_FuseCode.FUSED_NONE, transpose=True
        ),
        "quantized::add": lambda self, node: self.add_qadd(
            node, NNAPI_OperationCode.ADD, NNAPI_FuseCode.FUSED_NONE
        ),
        "quantized::add_relu": lambda self, node: self.add_qadd(
            node, NNAPI_OperationCode.ADD, NNAPI_FuseCode.FUSED_RELU
        ),
        "quantized::mul": lambda self, node: self.add_qadd(
            node, NNAPI_OperationCode.MUL, NNAPI_FuseCode.FUSED_NONE
        ),
    }

    def add_node(self, node):
        adder = self.ADDER_MAP.get(node.kind())
        if not adder:
            raise Exception(  # noqa: TRY002
                f"Unsupported node kind ({node.kind()!r}) in node {node!r}"
            )
        adder(self, node)

    def _identity(self, node):
        in_id, _in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        jitval = node.outputsAt(0)
        self.jitval_operand_map[jitval] = in_id

    def add_getattr(self, node):
        if node.inputsSize() != 1:
            raise AssertionError(
                f"expected node.inputsSize() == 1, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        obj_ctype, obj = self.get_constant_value(node.inputsAt(0))
        if not str(obj_ctype).startswith("__torch__."):
            raise AssertionError(
                f"expected obj_ctype to start with '__torch__.', got {obj_ctype}"
            )
        name = node.s("name")
        value = getattr(obj, name)
        output = node.outputsAt(0)
        ctype = output.type()
        self.add_constant_value(output, ctype, value)

    def add_constant_node(self, node):
        if node.inputsSize() != 0:
            raise AssertionError(
                f"expected node.inputsSize() == 0, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        output = node.outputsAt(0)
        ctype = output.type()
        value = output.toIValue()
        self.add_constant_value(output, ctype, value)

    def add_list_construct(self, node):
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        output = node.outputsAt(0)
        ctype = output.type()
        const_vals: list | None = []
        tensors: list | None = []
        for inp in node.inputs():
            if const_vals is not None and inp in self.constants:
                _, val = self.get_constant_value(inp)
                const_vals.append(val)
            else:
                const_vals = None
            if tensors is not None and inp.type().kind() == "TensorType":
                tensors.append(inp)
            else:
                tensors = None

        if const_vals is not None:
            # NOTE: Now that TorchScript supports list constants,
            # this code path might not be used anymore.
            self.add_constant_value(output, ctype, const_vals)
        if tensors is not None:
            self.add_tensor_sequence(output, tensors)
        if const_vals is None and tensors is None:
            raise Exception(  # noqa: TRY002
                f"Unable to handle ListConstruct node.  Neither all constants nor all tensors. {node!r}"
            )

    def add_tuple_construct(self, node):
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        output = node.outputsAt(0)
        values = list(node.inputs())
        self.add_tensor_sequence(output, values)

    def add_unsqueeze(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))

        _, dim = self.get_constant_value(node.inputsAt(1), "IntType")
        if in_oper.dim_order != DimOrder.PRESUMED_CONTIGUOUS:
            raise AssertionError(
                f"expected dim_order PRESUMED_CONTIGUOUS, got {in_oper.dim_order}"
            )

        real_dim = dim if dim >= 0 else dim + len(in_oper.shape) + 1
        out_shape_list = list(in_oper.shape)
        out_shape_list.insert(real_dim, 1)
        out_shape = tuple(out_shape_list)
        out_oper = in_oper._replace(shape=out_shape)

        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_scalar(dim)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.EXPAND_DIMS, inputs, outputs)

    def add_to(self, node):
        # Handle to("cpu") / to("gpu") case
        self._identity(node)

    def add_reshape(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))

        shape_ctype, shape = self.get_constant_value(node.inputsAt(1))
        if shape_ctype.kind() != "ListType":
            raise AssertionError(
                f"expected shape_ctype ListType, got {shape_ctype.kind()}"
            )
        if shape_ctype.getElementType().kind() != "IntType":
            raise AssertionError(
                f"expected shape element type IntType, got {shape_ctype.getElementType().kind()}"
            )
        is_trivial_reshape = len(shape) == 2 and shape[1] == -1

        if in_oper.dim_order != DimOrder.PRESUMED_CONTIGUOUS and not is_trivial_reshape:
            raise Exception(  # noqa: TRY002
                "Currently, reshape is only supported on NHWC tensors if the target size is [X, -1]."
            )

        # Bit of a hack here.  Use a real tensor to infer the output shape.
        out_shape = torch.zeros(1).expand(in_oper.shape).reshape(shape).shape
        out_oper = in_oper._replace(
            shape=out_shape, dim_order=DimOrder.PRESUMED_CONTIGUOUS
        )

        inputs = [None] * 2
        inputs[0] = in_id
