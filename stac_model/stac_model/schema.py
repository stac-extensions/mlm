from pydantic import BaseModel
from .input import ModelInput, InputArray, Band, Statistics
from .output import ModelOutput, ClassObject, ResultArray, TaskEnum
from .runtime import Runtime, Asset, Container
from typing import List, Dict, Union

class MLModel(BaseModel):
    mlm_name: str
    mlm_task: TaskEnum
    mlm_framework: str
    mlm_framework_version: str
    mlm_file_size: int
    mlm_memory_size: int
    mlm_input: List[ModelInput]
    mlm_output: List[ModelOutput]
    mlm_runtime: List[Runtime]
    mlm_total_parameters: int
    mlm_pretrained_source: str
    mlm_summary: str
    mlm_parameters: Dict[str, Union[int, str, bool, List[Union[int, str, bool]]]] = None

__all__ = ["MLModel", "ModelInput", "InputArray", "Band", "Statistics", "ModelOutput", "ClassObject", "Asset", "ResultArray", "Runtime", "Container", "Asset"]
