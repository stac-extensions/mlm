from pydantic import BaseModel
from .input import ModelInput, InputArray, Band, Statistics
from .output import ModelOutput, ClassMap, ResultArray
from .runtime import Runtime, Asset, Container
from typing import List, Optional, Dict, Union

class Architecture(BaseModel):
    name: str
    file_size: int
    memory_size: int
    summary: str = None
    pretrained_source: str = None
    total_parameters: Optional[int] = None

class MLModel(BaseModel):
    mlm_name: str
    mlm_input: List[ModelInput]
    mlm_architecture: Architecture
    mlm_runtime: Runtime
    mlm_output: ModelOutput
    mlm_parameters: Dict[str, Union[int, str, bool, List[Union[int, str, bool]]]] = None

__all__ = ["MLModel", "ModelInput", "InputArray", "Band", "Statistics", "ModelOutput", "Asset", "ClassMap", "ResultArray", "Runtime", "Container", "Asset", "Architecture"]
