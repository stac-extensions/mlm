from .paths import S3Path
from pydantic import BaseModel, FilePath, field_validator
from typing import Optional, List
from enum import Enum
class Asset(BaseModel):
    """Information about the model location and other additional file locations. Follows
    the Asset Object spec: https://github.com/radiantearth/stac-spec/blob/master/collection-spec/collection-spec.md#asset-object
    """

    href: S3Path | FilePath | str
    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    roles: Optional[List[str]] = None


    class Config:
        arbitrary_types_allowed = True

    @field_validator("href")
    @classmethod
    def check_path_type(cls, v):
        if isinstance(v, str):
            if v.startswith("s3://"):
                v = S3Path(url=v)
            else:
                v = FilePath(f=v)
        else:
            raise ValueError(
                f"Expected str, S3Path, or FilePath input, received {type(v).__name__}"
            )
        return v

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
    asset: Asset
    source_code: Asset
    accelerator: AcceleratorEnum
    accelerator_constrained: bool
    hardware_summary: str
    container: Optional[Container] = None
    commit_hash: Optional[str] = None
    batch_size_suggestion: Optional[int] = None
