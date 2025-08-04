import glob
import inspect
import os
import pathlib
import tempfile
import zipfile
from collections.abc import Sequence

import torch

from ..base import Path
from .base import AOTIFiles


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


def aoti_compile_and_extract(
    program: torch.export.ExportedProgram,
    output_directory: Path
) -> list[Path]:
    """Compiles an exported program using AOTI and extracts the files to the specified directory.

    Args:
        program: The exported program to compile.
        output_directory: The directory where the compiled files will be extracted.

    Returns:
        A list of file paths extracted from the compiled package.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = torch._inductor.aoti_compile_and_package(
            program, package_path=os.path.join(tmpdir, "file.pt2")
        )

        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(output_directory)

    return [pathlib.Path(f) for f in glob.glob(os.path.join(output_directory, "**"), recursive=True) if os.path.isfile(f)]


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


def create_example_input_from_shape(input_shape: Sequence[int]) -> torch.Tensor:
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
