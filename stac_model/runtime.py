from enum import Enum
from typing import List, Optional

from pydantic import AnyUrl, BaseModel, ConfigDict, FilePath, Field


class Asset(BaseModel):
    """Information about the model location and other additional file locations.
    Follows the STAC Asset Object spec.
    """

    href: FilePath | AnyUrl | str
    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    roles: Optional[List[str]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Container(BaseModel):
    container_file: str
    image_name: str
    tag: str
    working_dir: str
    run: str
    accelerator: bool


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


class Runtime(BaseModel):
    framework: str
    framework_version: str
    file_size: int = Field(alias="file:size")
    memory_size: int
    batch_size_suggestion: Optional[int] = None

    accelerator: Optional[AcceleratorEnum] = Field(exclude_unset=True, default=None)
    accelerator_constrained: bool = Field(exclude_unset=True, default=False)
    accelerator_summary: str = Field(exclude_unset=True, exclude_defaults=True, default="")
    accelerator_count: int = Field(minimum=1, exclude_unset=True, exclude_defaults=True, default=-1)
