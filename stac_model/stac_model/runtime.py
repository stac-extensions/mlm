from .paths import S3Path
from pydantic import BaseModel, Field, FilePath, AnyUrl, field_validator
from typing import Optional, List
class ModelAsset(BaseModel):
    """Information about the model location and other additional file locations. Follows
    the Asset Object spec: https://github.com/radiantearth/stac-spec/blob/master/collection-spec/collection-spec.md#asset-object
    """

    href: S3Path | FilePath | str = Field(...)
    title: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    type: Optional[str] = Field(None)
    roles: Optional[List[str]] = Field(None)


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

class ContainerInfo(BaseModel):
    container_file: str
    image_name: str
    tag: str
    working_dir: str
    run: str
    accelerator: bool

class Runtime(BaseModel):
    framework: str
    version: str
    model_asset: ModelAsset
    model_handler: str
    model_src_url: str
    model_commit_hash: str
    container: List[ContainerInfo]
    batch_size_suggestion: int
    hardware_suggestion: str | AnyUrl
