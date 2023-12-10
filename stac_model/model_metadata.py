from pydantic import BaseModel, Field, FilePath, AnyUrl
from typing import Optional, List, Tuple, Dict, Literal, Any
from uuid import uuid4
import numpy as np
import re

# Pydantic Models
class TensorSignature(BaseModel):
    name: Optional[str] = None
    dtype: Any = Field(...)
    shape: Tuple[int, ...] | List[int] = Field(...)

class ModelSignature(BaseModel):
    inputs: List[TensorSignature]
    outputs: List[TensorSignature]
    params: Optional[Dict[str, int | float | str]] = None

    class Config:
        arbitrary_types_allowed = True

class RuntimeConfig(BaseModel):
    environment: str

class S3Path(AnyUrl):
    allowed_schemes = {'s3'}
    user_required = False
    max_length = 1023
    min_length = 8

    @classmethod
    def validate_s3_url(cls, v):
        if not v.startswith('s3://'):
            raise ValueError('S3 path must start with s3://')
        return v

    @classmethod
    def validate_bucket_name(cls, v):
        if not v:
            raise ValueError('Bucket name cannot be empty')
        return v

    @classmethod
    def validate_key(cls, v):
        if '//' in v:
            raise ValueError('Key must not contain double slashes')
        return v.strip('/')

class ModelArtifact(BaseModel):
    path: S3Path | FilePath | str = Field(...)
    additional_files: Optional[Dict[str, FilePath]] = None

    class Config:
        arbitrary_types_allowed = True

class ClassMap(BaseModel):
    class_to_label_id: Dict[str, int]

    @property
    def label_id_to_class(self) -> Dict[int, str]:
        return {v: k for k, v in self.class_to_label_id.items()}

class ModelMetadata(BaseModel):
    signatures: ModelSignature
    artifact: ModelArtifact
    id: str = Field(default_factory=lambda: uuid4().hex)
    class_map: ClassMap
    runtime_config: Optional[RuntimeConfig] = None
    name: str
    ml_model_type: Optional[str] = None
    ml_model_processor_type: Optional[Literal["cpu", "gpu", "tpu", "mps"]] = None
    ml_model_learning_approach: Optional[str] = None
    ml_model_prediction_type: Optional[Literal["object-detection", "classification", "segmentation", "regression"]] = None
    ml_model_architecture: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

# Functions to create, serialize, and deserialize ModelMetadata
def create_metadata():
    input_sig = TensorSignature(name='input_tensor', dtype='float32', shape=(-1, 13, 64, 64))
    output_sig = TensorSignature(name='output_tensor', dtype='float32', shape=(-1, 10))
    model_sig = ModelSignature(inputs=[input_sig], outputs=[output_sig])
    model_artifact = ModelArtifact(path="s3://example/s3/uri/model.pt")
    class_map = ClassMap(class_to_label_id={
        'Annual Crop': 0, 'Forest': 1, 'Herbaceous Vegetation': 2, 'Highway': 3,
        'Industrial Buildings': 4, 'Pasture': 5, 'Permanent Crop': 6,
        'Residential Buildings': 7, 'River': 8, 'SeaLake': 9
    })
    return ModelMetadata(name="eurosat", class_map=class_map, signatures=model_sig, artifact=model_artifact, ml_model_processor_type="cpu")

def metadata_json(metadata: ModelMetadata) -> str:
    return metadata.model_dump_json(indent=2)

def model_metadata_json_operations(json_str: str) -> ModelMetadata:
    return ModelMetadata.model_validate_json(json_str)

# Running the functions end-to-end
metadata = create_metadata()
json_str = metadata_json(metadata)
model_metadata = model_metadata_json_operations(json_str)

print("Model Metadata Name:", model_metadata.name)
