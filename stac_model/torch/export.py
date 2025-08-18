import logging
import pathlib
import tempfile

import torch
from torch.export.dynamic_shapes import Dim
from torch.export.pt2_archive._package import package_pt2

from ..base import Path
from ..schema import MLModelProperties
from .base import AOTIFiles, ExportedPrograms, ExtraFiles
from .utils import (
    aoti_compile,
    create_example_input_from_shape,
    extract_module_arg_names,
    model_properties_to_metadata,
    update_properties,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def export(
    input_shape: list[int],
    model: torch.nn.Module,
    transforms: torch.nn.Module | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.export.ExportedProgram, torch.export.ExportedProgram | None]:
    """Exports a model and its transforms to programs.

    Args:
        model: The model to export.
        transforms: The transforms to export. The transforms should be a `torch.nn.Module. If you have
            multiple transforms, it is recommended to wrap them in a `torch.nn.Sequential`.
        input_shape: The shape of the input tensor, where -1 indicates a dynamic dimension.
        device: The device to export the model and transforms to.
        dtype: The data type to use for the model and transforms. Defaults to torch.float32.

    Returns:
        The exported model and transforms programs.
    """
    example_inputs = create_example_input_from_shape(input_shape).to(device).to(dtype)
    dims = tuple(Dim.AUTO if dim == -1 else dim for dim in input_shape)
    logger.debug("Exporting with dims: %s", dims)
    logger.debug("Example input shape: %s", list(example_inputs.shape))

    model.eval()
    model = model.to(device).to(dtype)
    model_arg = extract_module_arg_names(model)
    model_program = torch.export.export(mod=model, args=(example_inputs,), dynamic_shapes={model_arg: dims})

    if transforms is not None:
        transforms.eval()
        transforms = transforms.to(device).to(dtype)
        transforms_arg = extract_module_arg_names(transforms)
        transforms_program = torch.export.export(
            mod=transforms, args=(example_inputs,), dynamic_shapes={transforms_arg: dims}
        )
    else:
        transforms_program = None

    return model_program, transforms_program


def package(
    output_file: Path,
    model_program: torch.export.ExportedProgram,
    transforms_program: torch.export.ExportedProgram | None = None,
    metadata_properties: MLModelProperties | None = None,
    aoti_compile_and_package: bool = False,
) -> None:
    """Packages a model and its transforms AOTI exported programs into a single archive file.

    Args:
        output_file: The path to the output archive file.
        model_program: The exported model program.
        transforms_program: The exported transforms program.
        metadata_properties: MLModelProperties object
        aoti_compile_and_package: Whether to compile and package the model and transforms using AOTI.

    Raises:
        ValidationError: if the model metadata is not valid MLModelProperties.
    """
    aoti_files: AOTIFiles = {}
    extra_files: ExtraFiles = {}
    exported_programs: ExportedPrograms = {}

    if metadata_properties is not None:
        metadata_yaml = model_properties_to_metadata(metadata_properties)
        extra_files["mlm-metadata"] = metadata_yaml

    if aoti_compile_and_package:
        model_tmpdir = tempfile.TemporaryDirectory()
        transforms_tmpdir = tempfile.TemporaryDirectory()
        aoti_files.update(
            aoti_compile(
                model_directory=pathlib.Path(model_tmpdir.name),
                model_program=model_program,
                transforms_directory=pathlib.Path(transforms_tmpdir.name),
                transforms_program=transforms_program,
            )
        )
    else:
        exported_programs["model"] = model_program
        if transforms_program is not None:
            exported_programs["transforms"] = transforms_program

    package_pt2(
        f=output_file,
        exported_programs=exported_programs or None,  # type: ignore[arg-type]
        aoti_files=aoti_files or None,  # type: ignore[arg-type]
        extra_files=extra_files or None,  # type: ignore[arg-type]
    )

    if aoti_compile_and_package:
        model_tmpdir.cleanup()
        transforms_tmpdir.cleanup()


def save(
    output_file: Path,
    input_shape: list[int],
    model: torch.nn.Module,
    transforms: torch.nn.Module | None = None,
    metadata: Path | MLModelProperties | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    aoti_compile_and_package: bool = False,
) -> None:
    """Exports a model and its transforms to a packaged archive file.
    Args:
        output_file: The path to the output archive file.
        input_shape: The shape of the input tensor, where -1 indicates a dynamic dimension.
        model: The model to export.
        transforms: The transforms to export. The transforms should be a `torch.nn.Module`. If you have
            multiple transforms, it is recommended to wrap them in a `torch.nn.Sequential`.
        metadata: Path to the YAML file containing model metadata or an instance of MLModelProperties.
        device: The device to export the model and transforms to.
        dtype: The data type to use for the model and transforms. Defaults to torch.float32.
        aoti_compile_and_package: Whether to compile and package the model and transforms using AOTI.
    """
    model_program, transforms_program = export(
        model=model,
        transforms=transforms,
        input_shape=input_shape,
        device=device,
        dtype=dtype,
    )

    if metadata is not None:
        metadata_properties = update_properties(metadata=metadata, input_shape=input_shape, device=device, dtype=dtype)

    package(
        output_file=output_file,
        model_program=model_program,
        transforms_program=transforms_program,
        metadata_properties=metadata_properties,
        aoti_compile_and_package=aoti_compile_and_package,
    )
