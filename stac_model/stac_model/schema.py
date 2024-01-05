from pydantic import BaseModel
from .input import ModelInput, Array, Band
from .output import ModelOutput, ClassMap
from .runtime import Runtime, ModelAsset

class Architecture(BaseModel):
    total_parameters: int
    on_disk_size_mb: float
    ram_size_mb: float
    model_type: str
    summary: str
    pretrained: str

class MLModel(BaseModel):
    mlm_input: ModelInput
    mlm_architecture: Architecture
    mlm_runtime: Runtime
    mlm_output: ModelOutput

__all__ = ["MLModel", "ModelInput", "Array", "Band", "ModelOutput", "ModelAsset", "ClassMap", "Runtime", "Architecture"]
