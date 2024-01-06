from pydantic import BaseModel
from typing import Optional
from .input import Input, InputArray, Band, Statistics
from .output import Output, ClassMap
from .runtime import Runtime, Asset, ContainerInfo

class Architecture(BaseModel):
    name: str
    summary: str
    pretrained: bool
    total_parameters: Optional[int] = None
    on_disk_size_mb: Optional[float] = None
    ram_size_mb: Optional[float] = None

class MLModel(BaseModel):
    mlm_input: Input
    mlm_architecture: Architecture
    mlm_runtime: Runtime
    mlm_output: Output

__all__ = ["MLModel", "Input", "InputArray", "Band", "Statistics", "Output", "Asset", "ClassMap", "Runtime", "ContainerInfo", "Asset", "Architecture"]
