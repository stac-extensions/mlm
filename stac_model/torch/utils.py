import glob
import inspect
import os
import pathlib
import tempfile
import zipfile
from typing import Any, cast

import kornia.augmentation as K
import torch
import yaml

from ..base import DataType, Path
from ..runtime import AcceleratorName
from ..schema import SCHEMA_URI, MLModelProperties
from .base import AOTIFiles


def normalize_dtype(torch_dtype: torch.dtype) -> DataType:
    """
    Convert a PyTorch dtype (e.g., torch.float32) to a standardized DataType.
    """
    return cast(DataType, str(torch_dtype).rsplit(".", 1)[-1])


def find_tensor_by_key(state_dict: dict[str, torch.Tensor], key_substring: str, reverse: bool = False) -> torch.Tensor:
    """
    Find a tensor in the state_dict by a substring in its key.
    If `reverse` is True, search from the end of the dictionary.
    """
    items = reversed(state_dict.items()) if reverse else state_dict.items()
    for key, tensor in items:
        if key_substring in key:
            return tensor
    raise ValueError(f"Could not find tensor with key containing '{key_substring}'")


def get_input_hw(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    tensor = find_tensor_by_key(state_dict, "encoder._conv_stem.weight")
    return tensor.shape[2], tensor.shape[3]


def get_input_dtype(state_dict: dict[str, torch.Tensor]) -> DataType:
    """
    Get the data type (dtype) of the input from the first convolutional layer's weights.
    """
    tensor = find_tensor_by_key(state_dict, "encoder._conv_stem.weight")
    return normalize_dtype(tensor.dtype)


def get_output_dtype(state_dict: dict[str, torch.Tensor]) -> DataType:
    """
    Get the data type (dtype) of the output from the segmentation head's last conv layer.
    """
    tensor = find_tensor_by_key(state_dict, "segmentation_head.0.weight", reverse=True)
    return normalize_dtype(tensor.dtype)


def get_input_channels(state_dict: dict[str, torch.Tensor]) -> int:
    """
    Get number of input channels from the first convolutional layer's weights.
    """
    tensor = find_tensor_by_key(state_dict, "encoder._conv_stem.weight")
    return int(tensor.shape[1])


def get_output_channels(state_dict: dict[str, torch.Tensor]) -> int:
    """
    Get number of output channels from the segmentation head's last conv layer.
    """
    tensor = find_tensor_by_key(state_dict, "segmentation_head.0.weight", reverse=True)
    return int(tensor.shape[0])


def extract_value_scaling(transforms: K.AugmentationSequential) -> list[dict[str, Any]]:
    children = list(transforms.children())

    def _tensor_to_value(tensor) -> Any:
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()

    scaling_defs = []

    for t in children:
        if isinstance(t, K.Normalize):
            buffers = dict(t.named_buffers()) if hasattr(t, "named_buffers") else {}
            mean = buffers.get("mean")
            stddev = buffers.get("std")

            if mean is None or stddev is None:
                flags = getattr(t, "flags", {})
                mean = mean or flags.get("mean")
                stddev = stddev or flags.get("std")

            if mean is None or stddev is None:
                raise AttributeError("Normalize transform missing mean/std info")

            scaling_defs.append(
                {
                    "type": "z-score",
                    "mean": int(_tensor_to_value(mean)),
                    "stddev": int(_tensor_to_value(stddev)),
                }
            )

        elif isinstance(t, K.AugmentationSequential):
            scaling_defs.extend(extract_value_scaling(t))

    return scaling_defs


def extract_module_arg_names(module: torch.nn.Module) -> str:
    """Extracts the argument names of the forward method of a given module.

    Args:
        module: The PyTorch module from which to extract argument names.

    Returns:
        A list of argument names for the forward method of the module.

    Raises:
        ValueError: If the module does not have a forward method.
    """
    if not hasattr(module, "forward"):
        raise ValueError("The provided module does not have a forward method.")

    return next(iter(inspect.signature(module.forward).parameters))


def aoti_compile_and_extract(program: torch.export.ExportedProgram, output_directory: Path) -> list[Path]:
    """Compiles an exported program using AOTI and extracts the files to the specified directory.

    Args:
        program: The exported program to compile.
        output_directory: The directory where the compiled files will be extracted.

    Returns:
        A list of file paths extracted from the compiled package.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = torch._inductor.aoti_compile_and_package(program, package_path=os.path.join(tmpdir, "file.pt2"))

        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(output_directory)

    return [
        cast(os.PathLike[str], f)
        for f in glob.glob(os.path.join(output_directory, "**"), recursive=True)
        if os.path.isfile(f)
    ]


def aoti_compile(
    model_directory: Path,
    model_program: torch.export.ExportedProgram,
    transforms_directory: Path | None = None,
    transforms_program: torch.export.ExportedProgram | None = None,
) -> AOTIFiles:
    """Compiles a model and its transforms using AOTI.

    Args:
        model_directory: The directory to store the compiled model files.
        model_program: The exported model program.
        transforms_directory: The directory to store the compiled transforms files.
        transforms_program: The exported transforms program.
    """
    model_files = aoti_compile_and_extract(
        program=model_program,
        output_directory=model_directory,
    )
    aoti_files: AOTIFiles = {"model": model_files}

    if transforms_program is not None and transforms_directory is not None:
        transforms_files = aoti_compile_and_extract(
            program=transforms_program,
            output_directory=transforms_directory,
        )
        aoti_files["transforms"] = transforms_files

    return aoti_files


def create_example_input_from_shape(input_shape: list[int]) -> torch.Tensor:
    """Creates an example input tensor based on the provided input shape.

    If batch dimension is dynamic (-1), it defaults to 2. Other dynamic dimensions
    default to 224. Ideally all dimensions are defined by the user but this provides good defaults.

    Args:
        input_shape: The shape of the input tensor.

    Returns:
        A tensor filled with random values, shaped according to the input_shape.

    Raises:
        ValueError: If all dimensions are dynamic (-1).
    """
    shape = []

    if all(dim == -1 for dim in input_shape):
        raise ValueError("Input shape cannot be all dynamic (-1). At least one dimension must be fixed.")
    elif any(dim != -1 for dim in input_shape):
        batch_dim = 2 if input_shape[0] == -1 else input_shape[0]
        shape.append(batch_dim)
        shape.extend([dim if dim != -1 else 224 for dim in input_shape[1:]])
    else:
        shape = list(input_shape)

    return torch.randn(*shape, requires_grad=False)


def model_properties_to_metadata(properties: MLModelProperties) -> str:
    """Converts MLModelProperties to a metadata dictionary in YAML format.

    Args:
        properties: An instance of MLModelProperties containing model metadata.

    Returns:
        A YAML string representation of the model properties.
    """
    properties_dict = properties.model_dump(by_alias=False, exclude_none=True)
    properties_dict = {k.replace("mlm:", ""): v for k, v in properties_dict.items()}
    return yaml.dump(
        {
            "$schema": SCHEMA_URI,
            "properties": properties_dict,
        },
        default_flow_style=False,
    )


def update_properties(
    metadata: Path | MLModelProperties, input_shape: list[int], device: str | torch.device, dtype: torch.dtype
) -> MLModelProperties:
    """Updates the MLModelProperties with the given metadata, device, and dtype.

    Args:
        metadata: Path to the YAML file containing model metadata or an instance of MLModelProperties.
        input_shape: The shape of the input tensor, where -1 indicates a dynamic dimension.
        device: The device to export the model and transforms to.
        dtype: The data type to use for the model and transforms.

    Returns:
        An instance of MLModelProperties with updated properties.

    Raises:
        ValidationError: if the metadata is not valid MLModelProperties.
        TypeError: if metadata is not a path to a YAML file or an instance of MLModelProperties.
    """
    if isinstance(metadata, pathlib.Path | str):
        with open(metadata) as f:
            meta = yaml.safe_load(f)
            properties = MLModelProperties(**meta["properties"])
    elif isinstance(metadata, MLModelProperties):
        properties = metadata
    else:
        raise TypeError("Metadata must be a path to a YAML file or an instance of MLModelProperties.")

    accelerator = cast(AcceleratorName, str(device).split(":")[0])
    data_type = cast(DataType, str(dtype).split(".")[-1])

    properties.accelerator = accelerator
    properties.input[0].input.shape = input_shape  # type: ignore[assignment]
    properties.input[0].input.data_type = data_type
    properties.output[0].result.data_type = data_type

    return properties
