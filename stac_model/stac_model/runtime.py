from .paths import S3Path
from pydantic import BaseModel, Field, FilePath, field_validator
from typing import Optional, Dict
class ModelArtifact(BaseModel):
    """Information about the model location and other additional file locations."""

    path: S3Path | FilePath | str = Field(...)
    additional_files: Optional[Dict[str, FilePath]] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator("path")
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
