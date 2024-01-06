from pydantic import BaseModel
from typing import Optional
from .input import ModelInput, InputArray, Band, Statistics
from .output import ModelOutput, ClassMap
from .runtime import Runtime, ModelAsset

class Architecture(BaseModel):
    name: str
    model_type: str
    summary: str
    pretrained: bool
    total_parameters: Optional[int]
    on_disk_size_mb: Optional[float]
    ram_size_mb: Optional[float]

class MLModel(BaseModel):
    mlm_input: ModelInput
    mlm_architecture: Architecture
    mlm_runtime: Runtime
    mlm_output: ModelOutput

__all__ = ["MLModel", "ModelInput", "InputArray", "Band", "Statistics", "ModelOutput", "ModelAsset", "ClassMap", "Runtime", "ContainerInfo", "Model Asset", "Architecture"]
