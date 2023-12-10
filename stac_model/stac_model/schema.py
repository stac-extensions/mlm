from pydantic import (
    BaseModel,
    Field,
    FilePath,
    field_validator,
    field_serializer,
    AnyUrl,
    ConfigDict,
)
from typing import Optional
import re
from typing import List, Tuple, Dict, Optional, Literal, Any
import numpy as np
from uuid import uuid4
# import numpy.typing as npt


class TensorSignature(BaseModel):
    """Tensor metadata, including the dtype (int8, float32, etc) and the tensor shape."""

    name: Optional[str] = None
    # TODO there's a couple of issues blocking numpy typing with
    # pydantic or I'm not familiar enough with custom validators
    # https://github.com/numpy/numpy/issues/25206
    # dtype: npt.DTypeLike = Field(...)
    dtype: str = Field(...)
    shape: Tuple[int, ...] | List[int] = Field(...)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # TODO can't take numpy types for now until new pydant 2.6
    # # NotImplementedError: Cannot check isinstance when validating from json, use a JsonOrPython validator instead.
    # @field_serializer('dtype')
    # def serialize_ndtype(self, dtype: np.dtype) -> str:
    #     return dtype.name
    # @field_validator('dtype', mode="before")
    # @classmethod
    # def convert_dtype(cls, v):
    #     if isinstance(v, str):
    #         v = np.dtype(v)
    #     elif not isinstance(v, np.dtype):
    #         raise ValueError(f'Expected np.dtype, received {type(v).__name__}')
    #     return v

    # @field_validator('shape')
    # @classmethod
    # def validate_shape(cls, v):
    #     if not isinstance(v, (tuple, list)):
    #         raise ValueError(f'Expected tuple or list for shape, received {type(v).__name__}')
    #     return list(v)


class ModelSignature(BaseModel):
    """The name of the input tensor and accompanying tensor metadata."""

    inputs: List[TensorSignature]
    outputs: List[TensorSignature]
    params: Optional[
        Dict[str, int | float | str]
    ] = None  # Or any other type that 'params' might take

    class Config:
        arbitrary_types_allowed = True

    @property
    def inputs_length(self) -> int:
        return len(self.inputs)

    @property
    def outputs_length(self) -> int:
        return len(self.outputs)


class RuntimeConfig(BaseModel):
    """TODO decide how to handle model runtime configurations. dependencies and hyperparams"""

    environment: str


class S3Path(AnyUrl):
    allowed_schemes = {"s3"}
    user_required = False
    max_length = 1023
    min_length = 8

    @field_validator("url")
    @classmethod
    def validate_s3_url(cls, v):
        if not v.startswith("s3://"):
            raise ValueError("S3 path must start with s3://")
        if len(v) < cls.min_length:
            raise ValueError("S3 path is too short")
        if len(v) > cls.max_length:
            raise ValueError("S3 path is too long")
        return v

    @field_validator("host")
    @classmethod
    def validate_bucket_name(cls, v):
        if not v:
            raise ValueError("Bucket name cannot be empty")
        if not 3 <= len(v) <= 63:
            raise ValueError("Bucket name must be between 3 and 63 characters")
        if not re.match(r"^[a-z0-9.\-]+$", v):
            raise ValueError(
                "Bucket name can only contain lowercase letters, numbers, dots, and hyphens"
            )
        if v.startswith("-") or v.endswith("-"):
            raise ValueError("Bucket name cannot start or end with a hyphen")
        if ".." in v:
            raise ValueError("Bucket name cannot have consecutive periods")
        return v

    @field_validator("path")
    @classmethod
    def validate_key(cls, v):
        if "//" in v:
            raise ValueError("Key must not contain double slashes")
        if "\\" in v:
            raise ValueError("Backslashes are not standard in S3 paths")
        if "\t" in v or "\n" in v:
            raise ValueError("Key cannot contain tab or newline characters")
        return v.strip("/")


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


class ClassMap(BaseModel):
    class_to_label_id: Dict[str, int]

    # Property to reverse the mapping
    @property
    def label_id_to_class(self) -> Dict[int, str]:
        # Reverse the mapping
        return {v: k for k, v in self.class_to_label_id.items()}

    def get_class(self, class_id: int) -> str:
        """Get class name from class id."""
        if class_id not in self.label_id_to_class:
            raise ValueError(f"Class ID '{class_id}' not found")
        return self.label_id_to_class[class_id]

    def get_label_id(self, class_name: str) -> int:
        """Get class id from class name."""
        if class_name not in self.class_to_label_id:
            raise ValueError(f"Class name '{class_name}' not found")
        return self.class_to_label_id[class_name]


class ModelMetadata(BaseModel):
    signatures: ModelSignature
    artifact: ModelArtifact
    id: str = Field(default_factory=lambda: uuid4().hex)
    class_map: ClassMap

    # Runtime configurations required to run the model.
    # TODO requirements.txt , conda.yml, or lock files for each should be supported in future.
    runtime_config: Optional[RuntimeConfig] = None

    # the name of the model
    name: str
    ml_model_type: Optional[str] = None
    ml_model_processor_type: Optional[Literal["cpu", "gpu", "tpu", "mps"]] = None
    ml_model_learning_approach: Optional[str] = None
    ml_model_prediction_type: Optional[
        Literal["object-detection", "classification", "segmentation", "regression"]
    ] = None
    ml_model_architecture: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
