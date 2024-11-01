from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import AliasChoices, Field

from stac_model.base import MLMBaseModel, OmitIfNone


class AcceleratorEnum(str, Enum):
    amd64 = "amd64"
    cuda = "cuda"
    xla = "xla"
    amd_rocm = "amd-rocm"
    intel_ipex_cpu = "intel-ipex-cpu"
    intel_ipex_gpu = "intel-ipex-gpu"
    macos_arm = "macos-arm"

    def __str__(self):
        return self.value


AcceleratorName = Literal[
    "amd64",
    "cuda",
    "xla",
    "amd-rocm",
    "intel-ipex-cpu",
    "intel-ipex-gpu",
    "macos-arm",
]

AcceleratorType = Union[AcceleratorName, AcceleratorEnum]


class Runtime(MLMBaseModel):
    framework: Annotated[str | None, OmitIfNone] = Field(default=None)
    framework_version: Annotated[str | None, OmitIfNone] = Field(default=None)
    file_size: Annotated[int | None, OmitIfNone] = Field(
        alias="file:size",
        validation_alias=AliasChoices("file_size", "file:size"),
        default=None,
    )
    memory_size: Annotated[int | None, OmitIfNone] = Field(default=None)
    batch_size_suggestion: Annotated[int | None, OmitIfNone] = Field(default=None)

    accelerator: AcceleratorType | None = Field(default=None)
    accelerator_constrained: bool = Field(default=False)
    accelerator_summary: Annotated[str | None, OmitIfNone] = Field(default=None)
    accelerator_count: Annotated[int | None, OmitIfNone] = Field(default=None, minimum=1)
