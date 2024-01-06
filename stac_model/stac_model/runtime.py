from .paths import S3Path
from pydantic import BaseModel, FilePath, AnyUrl, field_validator
from typing import Optional, List
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
    asset: Asset
    source_code_url: str
    handler: Optional[str] = None
    commit_hash: Optional[str] = None
    container: Optional[ContainerInfo] = None
    batch_size_suggestion: Optional[int] = None
    hardware_suggestion: Optional[str | AnyUrl] = None
