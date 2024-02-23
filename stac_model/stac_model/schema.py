from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict

from .input import Band, InputArray, ModelInput, Statistics
from .output import ClassObject, ModelOutput, ResultArray, TaskEnum
from .runtime import Asset, Container, Runtime


def mlm_prefix_replacer(field_name: str) -> str:
    return field_name.replace("mlm_", "mlm:")

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
    mlm_parameters: Dict[str, Union[int, str, bool, List[Union[int, str, bool]]]] = None # noqa: E501

    model_config = ConfigDict(alias_generator=mlm_prefix_replacer,
                              populate_by_name=True, extra="ignore")


__all__ = [
    "MLModel",
    "ModelInput",
    "InputArray",
    "Band",
    "Statistics",
    "ModelOutput",
    "ClassObject",
    "Asset",
    "ResultArray",
    "Runtime",
    "Container",
    "Asset",
]
