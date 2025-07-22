import os
from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2 as T
import yaml
from torch.export.pt2_archive._package import load_pt2
from torchgeo.models import Unet_Weights, unet

from stac_model.torch.export import (
    export,
    package,
)

pytest.importorskip("torch")


def export_model(tmpdir: Path, device: str, aoti_compile_and_package: bool) -> None:
    device = torch.device(device)
    input_shape = (-1, 8, -1, -1)
    archive_path = os.path.join(tmpdir, "model.pt2")
    metadata_path = os.path.join("tests", "torch", "ftw-metadata.yaml")
    weights = Unet_Weights.SENTINEL2_3CLASS_FTW
    transforms = torch.nn.Sequential(*[T.Resize((256, 256)), T.Normalize(mean=[0.0], std=[3000.0])])
    model = unet(weights=weights)
    model_program, transforms_program = export(
        model=model,
        transforms=transforms,
        input_shape=input_shape,
        device=device,
    )
    package(
        output_file=archive_path,
        model_program=model_program,
        transforms_program=transforms_program,
        metadata_path=metadata_path,
        aoti_compile_and_package=aoti_compile_and_package,
    )

    # Validate that pt2 is loadable
    pt2 = load_pt2(archive_path)
    if pt2.aoti_runners == {}:
        model = pt2.aoti_runners["model"]
        transforms = pt2.aoti_runners["transforms"]
    else:
        model = pt2.exported_programs["model"].module()
        transforms = pt2.exported_programs["transforms"].module()

    # Validate transforms are usable
    x = torch.randn(1, 8, 128, 128, device=device, requires_grad=False)
    transforms(x)

    # Validate model is usable
    x = torch.randn(1, 8, 128, 128, device=device, requires_grad=False)
    model(x)

    # Validate metadata is valid yaml
    metadata = pt2.extra_files["metadata"]
    yaml.safe_load(metadata)


@pytest.mark.parametrize("aoti_compile_and_package", [True, False])
def test_ftw_export_cpu(tmpdir: Path, aoti_compile_and_package: bool) -> None:
    export_model(tmpdir, "cpu", aoti_compile_and_package=aoti_compile_and_package)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("aoti_compile_and_package", [True, False])
def test_ftw_export_cuda(tmpdir: Path, aoti_compile_and_package: bool) -> None:
    export_model(tmpdir, "cuda", aoti_compile_and_package=aoti_compile_and_package)
