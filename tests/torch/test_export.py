import pytest

pytest.importorskip("torchgeo")

import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
import yaml
from torch.export.pt2_archive._package import load_pt2
from torchgeo.models import Unet_Weights, unet

from stac_model.torch.export import (
    export,
    package,
)


def export_model(tmpdir: Path, device: str, aoti_compile_and_package: bool, no_transforms: bool) -> None:
    input_shape = (-1, 8, -1, -1)
    archive_path = os.path.join(tmpdir, "model.pt2")
    metadata_path = os.path.join("tests", "torch", "ftw-metadata.yaml")
    weights = Unet_Weights.SENTINEL2_3CLASS_FTW
    transforms = torch.nn.Sequential(T.Resize((256, 256)), T.Normalize(mean=[0.0], std=[3000.0]))
    model = unet(weights=weights)
    model_program, transforms_program = export(
        model=model,
        transforms=None if no_transforms else transforms,
        input_shape=input_shape,
        device=device,
        dtype=torch.float32,
    )
    package(
        output_file=archive_path,
        model_program=model_program,
        transforms_program=transforms_program,
        metadata_path=metadata_path,
        aoti_compile_and_package=aoti_compile_and_package,
    )

    # Validate that pt2 is loadable and model/transform are usable
    pt2 = load_pt2(archive_path)

    x = torch.randn(1, 8, 128, 128, device=device, dtype=torch.float32)
    if pt2.aoti_runners != {}:
        model_aoti = pt2.aoti_runners["model"]
        preds = model_aoti(x)
        assert preds.shape == (1, 3, 128, 128)

        if "transforms" in pt2.aoti_runners:
            transforms_aoti = pt2.aoti_runners["transforms"]
            transformed = transforms_aoti(x)
            assert transformed.shape == (1, 8, 256, 256)
    else:
        model_exported = pt2.exported_programs["model"].module()
        preds = model_exported(x)
        assert preds.shape == (1, 3, 128, 128)

        if "transforms" in pt2.exported_programs:
            transforms_exported = pt2.exported_programs["transforms"].module()
            transformed = transforms_exported(x)
            assert transformed.shape == (1, 8, 256, 256)

    # Validate metadata is valid yaml
    metadata = pt2.extra_files["metadata"]
    yaml.safe_load(metadata)


@pytest.mark.parametrize("no_transforms", [True, False])
@pytest.mark.parametrize("aoti_compile_and_package", [False, True])
def test_ftw_export_cpu(tmpdir: Path, aoti_compile_and_package: bool, no_transforms: bool) -> None:
    export_model(tmpdir, "cpu", aoti_compile_and_package, no_transforms)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("no_transforms", [True, False])
@pytest.mark.parametrize("aoti_compile_and_package", [False, True])
def test_ftw_export_cuda(tmpdir: Path, aoti_compile_and_package: bool, no_transforms: bool) -> None:
    export_model(tmpdir, "cuda", aoti_compile_and_package, no_transforms)
