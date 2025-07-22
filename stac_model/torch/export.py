import glob
import inspect
import logging
import os
import tempfile
import zipfile
from collections.abc import Sequence

import torch
import yaml
from torch.export.dynamic_shapes import Dim
from torch.export.pt2_archive._package import package_pt2

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def package(
    output_file: str,
    model_program: torch.export.ExportedProgram,
    transforms_program: torch.export.ExportedProgram | None = None,
    metadata_path: str | None = None,
    aoti_compile_and_package: bool = False,
) -> None:
    """Packages a model and its transforms AOTI exported programs into a single archive file.

    Args:
        output_file: The path to the output archive file.
        model_program: The exported model program.
        transforms_program: The exported transforms program.
        metadata_path: Path to the YAML file containing model metadata.
        aoti_compile_and_package: Whether to compile and package the model and transforms using AOTI. Defaults to True.
    """
    # Load metadata file
    if metadata_path is not None:
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        extra_files = {"metadata": yaml.dump(metadata)}
    else:
        extra_files = None

    # Package programs only
    if not aoti_compile_and_package:
        programs = {"model": model_program}
        if transforms_program is not None:
            programs["transforms"] = transforms_program
            package_pt2(
                f=output_file,
                exported_programs=programs,
                extra_files=extra_files,
            )

    # Attempt to AOTI compile and package
    else:
        with (
            tempfile.TemporaryDirectory() as archive_tmpdir,
            tempfile.TemporaryDirectory() as model_tmpdir,
            tempfile.TemporaryDirectory() as transforms_tmpdir,
        ):
            # Package and extract transforms from pt2 archive
            model_path = torch._inductor.aoti_compile_and_package(
                model_program, package_path=os.path.join(archive_tmpdir, "model.pt2")
            )

            # Extract model files from archive
            with zipfile.ZipFile(model_path, "r") as zip_ref:
                zip_ref.extractall(model_tmpdir)

            model_files = [
                f
                for f in glob.glob(os.path.join(model_tmpdir, "**"), recursive=True)
                if os.path.isfile(f)
            ]
            aoti_files = {"model": model_files}

            if transforms_program is not None:
                # Package and extract transforms from pt2 archive
                transforms_path = torch._inductor.aoti_compile_and_package(
                    transforms_program,
                    package_path=os.path.join(archive_tmpdir, "transforms.pt2"),
                )

                with zipfile.ZipFile(transforms_path, "r") as zip_ref:
                    zip_ref.extractall(transforms_tmpdir)

                transforms_files = [
                    f
                    for f in glob.glob(os.path.join(transforms_tmpdir, "**"), recursive=True)
                    if os.path.isfile(f)
                ]
                aoti_files["transforms"] = transforms_files

            # Package model, transforms, and metadata together
            package_pt2(
                f=output_file,
                aoti_files=aoti_files,
                extra_files=extra_files,
            )


@torch.no_grad()
def export(
    model: torch.nn.Module,
    transforms: torch.nn.Module,
    input_shape: Sequence[int],
    device: torch.device,
) -> tuple[torch.export.ExportedProgram, torch.export.ExportedProgram]:
    """Exports a model and its transforms to programs.

    Args:
        model: The model to export.
        transforms: The transforms to export.
        input_shape: The shape of the input tensor, where -1 indicates a dynamic dimension.
        device: The device to export the model and transforms to.

    Returns:
            The exported model and transforms programs.
    """
    model.eval()
    transforms.eval()
    model = model.to(device)
    transforms = transforms.to(device)

    # Construct example inputs and dims
    example_inputs = _create_example_input_from_shape(input_shape).to(device)
    dims = tuple(Dim.AUTO if dim == -1 else dim for dim in input_shape)
    logger.debug("Exporting with dims: {}", dims)
    logger.debug("Example input shape: {}", example_inputs.shape)

    # Extract argument names for the model and transforms forward methods
    model_arg = next(iter(inspect.signature(model.forward).parameters))
    transforms_arg = next(iter(inspect.signature(transforms.forward).parameters))

    # Export model and transforms
    model_program = torch.export.export(
        mod=model, args=(example_inputs,), dynamic_shapes={model_arg: dims}
    )
    transforms_program = torch.export.export(
        mod=transforms, args=(example_inputs,), dynamic_shapes={transforms_arg: dims}
    )
    return model_program, transforms_program


def _create_example_input_from_shape(input_shape: Sequence[int]) -> torch.Tensor:
    """Creates an example input tensor based on the provided input shape.

    Args:
        input_shape (Sequence[int]): The shape of the input tensor.

    Returns:
        torch.Tensor: A tensor filled with random values, shaped according to the input_shape.

    Raises:
        ValueError: If the channel dimension is dynamic (-1).
    """
    # Handle dynamic dimensions (-1) by replacing them with a fixed value
    shape = []

    # Batch dimension
    if input_shape[0] == -1:
        shape.append(2)
    else:
        shape.append(input_shape[0])

    # Channel dimension should always be set
    if input_shape[1] != -1:
        shape.append(input_shape[1])
    else:
        raise ValueError("Channel dimension cannot be dynamic (-1) for input shape.")

    # Height dimension
    if input_shape[2] == -1:
        shape.append(224)
    else:
        shape.append(input_shape[2])

    # Width dimension
    if input_shape[3] == -1:
        shape.append(224)
    else:
        shape.append(input_shape[3])

    return torch.randn(*shape, requires_grad=False)
