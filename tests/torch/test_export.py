import sys
import pytest

pytest.importorskip("torchgeo")
assert sys.version_info >= (3, 11), "torchgeo Unet requires Python 3.11+"

import pathlib

import torch
import torchvision.transforms.v2 as T
import yaml
from torch.export.pt2_archive._package import load_pt2
from torchgeo.models import Unet_Weights, unet

from stac_model.base import Path
from stac_model.schema import MLModelProperties
from stac_model.torch.export import save


class TestPT2:
    in_channels = 3
    num_classes = 2
    height = width = 16
    in_h = in_w = 8
    metadata_path = pathlib.Path("tests") / "torch" / "metadata.yaml"

    @pytest.fixture
    def model(self) -> torch.nn.Module:
        return torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_classes, kernel_size=1, padding=0)

    @pytest.fixture
    def transforms(self) -> torch.nn.Module:
        return torch.nn.Sequential(T.Resize((self.height, self.width)), T.Normalize(mean=[0.0], std=[255.0]))

    def validate(
        self,
        archive_path: pathlib.Path,
        no_transforms: bool,
        input_shape: list[int],
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Validate that pt2 is loadable and model/transform are usable."""
        pt2 = load_pt2(archive_path)

        x = torch.randn(1, self.in_channels, self.in_h, self.in_w, device=device, dtype=dtype)

        # Validate AOT Inductor saving
        if pt2.aoti_runners != {}:
            model_aoti = pt2.aoti_runners["model"]
            preds = model_aoti(x)
            assert preds.shape == (1, self.num_classes, self.in_h, self.in_w)

            if no_transforms:
                assert "transforms" not in pt2.aoti_runners
            else:
                assert "transforms" in pt2.aoti_runners

            if "transforms" in pt2.aoti_runners:
                transforms_aoti = pt2.aoti_runners["transforms"]
                transformed = transforms_aoti(x)
                assert transformed.shape == (1, self.in_channels, self.height, self.width)

        # Validate ExportedProgram saving
        else:
            model_exported = pt2.exported_programs["model"].module()
            preds = model_exported(x)
            assert preds.shape == (1, self.num_classes, self.in_h, self.in_w)

            if no_transforms:
                assert "transforms" not in pt2.exported_programs
            else:
                assert "transforms" in pt2.exported_programs

            if "transforms" in pt2.exported_programs:
                transforms_exported = pt2.exported_programs["transforms"].module()
                transformed = transforms_exported(x)
                assert transformed.shape == (1, self.in_channels, self.height, self.width)

        # Validate MLM model metadata
        metadata = pt2.extra_files["mlm-metadata"]
        metadata = yaml.safe_load(metadata)
        assert "mlm:accelerator" not in metadata["properties"]
        properties = MLModelProperties(**metadata["properties"])
        assert properties.input[0].input.shape == input_shape
        assert properties.accelerator == str(device).split(":")[0]
        assert properties.input[0].input.data_type == str(dtype).split(".")[-1]
        assert properties.output[0].result.data_type == str(dtype).split(".")[-1]

    @pytest.mark.parametrize("no_transforms", [True, False])
    @pytest.mark.parametrize("aoti_compile_and_package", [False, True])
    def test_export_model_cpu(
        self,
        tmpdir: Path,
        model: torch.nn.Module,
        transforms: torch.nn.Module,
        aoti_compile_and_package: bool,
        no_transforms: bool,
    ) -> None:
        archive_path = pathlib.Path(tmpdir) / "model.pt2"
        input_shape = [-1, self.in_channels, -1, -1]
        save(
            output_file=archive_path,
            model=model,
            transforms=None if no_transforms else transforms,
            metadata=self.metadata_path,
            input_shape=input_shape,
            device="cpu",
            dtype=torch.float32,
            aoti_compile_and_package=aoti_compile_and_package,
        )
        self.validate(
            archive_path=archive_path,
            no_transforms=no_transforms,
            input_shape=input_shape,
            device="cpu",
            dtype=torch.float32,
        )

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    @pytest.mark.parametrize("no_transforms", [True, False])
    @pytest.mark.parametrize("aoti_compile_and_package", [False, True])
    def test_export_model_cuda(
        self,
        tmpdir: Path,
        model: torch.nn.Module,
        transforms: torch.nn.Module,
        aoti_compile_and_package: bool,
        no_transforms: bool,
    ) -> None:
        archive_path = pathlib.Path(tmpdir) / "model.pt2"
        input_shape = [-1, self.in_channels, -1, -1]
        save(
            output_file=archive_path,
            model=model,
            transforms=None if no_transforms else transforms,
            metadata=self.metadata_path,
            input_shape=input_shape,
            device="cuda",
            dtype=torch.float32,
            aoti_compile_and_package=aoti_compile_and_package,
        )
        self.validate(
            archive_path=archive_path,
            no_transforms=no_transforms,
            input_shape=input_shape,
            device="cuda",
            dtype=torch.float32,
        )

    def test_export_mlmodelproperties(
        self,
        tmpdir: Path,
        model: torch.nn.Module,
    ) -> None:
        archive_path = pathlib.Path(tmpdir) / "model.pt2"
        input_shape = [-1, self.in_channels, -1, -1]

        with open(self.metadata_path) as f:
            metadata = yaml.safe_load(f)
            properties = MLModelProperties(**metadata["properties"])

        save(
            output_file=archive_path,
            model=model,
            transforms=None,
            metadata=properties,
            input_shape=input_shape,
            device=torch.device("cpu"),
            dtype=torch.float32,
            aoti_compile_and_package=False,
        )
        self.validate(
            archive_path=archive_path,
            no_transforms=True,
            input_shape=input_shape,
            device="cpu",
            dtype=torch.float32,
        )


class TestTorchGeoFTWPT2(TestPT2):
    in_channels = 8
    num_classes = 3
    height = width = 256
    in_h = in_w = 128
    metadata_path = pathlib.Path("tests") / "torch" / "metadata.yaml"

    @pytest.fixture
    def model(self) -> torch.nn.Module:
        model: torch.nn.Module = unet(weights=Unet_Weights.SENTINEL2_3CLASS_FTW)
        return model

    @pytest.fixture
    def transforms(self) -> torch.nn.Module:
        return torch.nn.Sequential(T.Resize((self.height, self.width)), T.Normalize(mean=[0.0], std=[3000.0]))

    @pytest.mark.slow
    @pytest.mark.parametrize("no_transforms", [True, False])
    @pytest.mark.parametrize("aoti_compile_and_package", [False, True])
    def test_export_model_cpu(
        self,
        tmpdir: Path,
        model: torch.nn.Module,
        transforms: torch.nn.Module,
        aoti_compile_and_package: bool,
        no_transforms: bool,
    ) -> None:
        archive_path = pathlib.Path(tmpdir) / "model.pt2"
        input_shape = [-1, self.in_channels, -1, -1]
        save(
            output_file=archive_path,
            model=model,
            transforms=None if no_transforms else transforms,
            metadata=self.metadata_path,
            input_shape=input_shape,
            device="cpu",
            dtype=torch.float32,
            aoti_compile_and_package=aoti_compile_and_package,
        )
        self.validate(
            archive_path=archive_path,
            no_transforms=no_transforms,
            input_shape=input_shape,
            device="cpu",
            dtype=torch.float32,
        )

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    @pytest.mark.parametrize("no_transforms", [True, False])
    @pytest.mark.parametrize("aoti_compile_and_package", [False, True])
    def test_export_model_cuda(
        self,
        tmpdir: Path,
        model: torch.nn.Module,
        transforms: torch.nn.Module,
        aoti_compile_and_package: bool,
        no_transforms: bool,
    ) -> None:
        archive_path = pathlib.Path(tmpdir) / "model.pt2"
        input_shape = [-1, self.in_channels, -1, -1]
        save(
            output_file=archive_path,
            model=model,
            transforms=None if no_transforms else transforms,
            metadata=self.metadata_path,
            input_shape=input_shape,
            device="cuda",
            dtype=torch.float32,
            aoti_compile_and_package=aoti_compile_and_package,
        )
        self.validate(
            archive_path=archive_path,
            no_transforms=no_transforms,
            input_shape=input_shape,
            device="cuda",
            dtype=torch.float32,
        )
